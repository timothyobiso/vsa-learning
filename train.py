"""Training loop for the scene encoder and amortized factorizer.

Modes:
  encoder    — Phase 1: train CNN+MLP encoder with cosine loss (default)
  factorizer — Phase 2: train factorizer MLP with frozen encoder, CE loss
  joint      — Phase 3: fine-tune both encoder and factorizer end-to-end
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from vsa.codebooks import SceneCodebooks
from vsa.resonator import ResonatorNetwork
from model.encoder import SceneEncoder
from model.factorizer import AmortizedFactorizer


def fhrr_cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative cosine similarity between complex vectors (to minimize).

    Args:
        pred: (B, d) complex
        target: (B, d) complex
    """
    # Normalize both
    pred_n = pred / pred.abs().clamp(min=1e-8)
    target_n = target / target.abs().clamp(min=1e-8)

    # Cosine similarity: real part of normalized dot product
    cos_sim = (pred_n * target_n.conj()).sum(dim=-1).real / pred.shape[-1]
    return -cos_sim.mean()


def collate_fn(batch):
    """Custom collate: stack images and targets, keep scene dicts as list."""
    images = torch.stack([b[0] for b in batch])
    targets = torch.stack([b[1] for b in batch])
    scenes = [b[2] for b in batch]
    return images, targets, scenes


# ── Target computation ──────────────────────────────────────────────────────

def compute_factor_targets(
    scenes: list[list[dict]], codebooks: SceneCodebooks, device: torch.device
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Convert a batch of scene dicts to per-factor target tensors.

    For single-object scenes, returns indices for the first object.
    For multi-object: returns indices for the first object (peeling target).

    Args:
        scenes: list of B scene dicts (each is a list of object dicts)
        codebooks: SceneCodebooks
        device: target device

    Returns:
        factor_targets: list of K tensors, each (B,) long
        stop_targets: (B,) float — 0 for real objects, 1 for stop
    """
    n_factors = len(codebooks.factors)
    B = len(scenes)

    factor_indices = [[] for _ in range(n_factors)]
    stop_targets = []

    for scene in scenes:
        indices = codebooks.object_to_indices(scene[0])
        for k in range(n_factors):
            factor_indices[k].append(indices[k])
        stop_targets.append(0.0)

    factor_targets = [
        torch.tensor(factor_indices[k], dtype=torch.long, device=device)
        for k in range(n_factors)
    ]
    stop_targets = torch.tensor(stop_targets, dtype=torch.float, device=device)
    return factor_targets, stop_targets


def factorizer_loss(
    factor_logits: list[torch.Tensor],
    stop_logit: torch.Tensor,
    factor_targets: list[torch.Tensor],
    stop_target: torch.Tensor,
) -> torch.Tensor:
    """Sum of per-factor cross-entropy + STOP BCE loss."""
    ce = nn.CrossEntropyLoss()
    bce = nn.BCEWithLogitsLoss()

    loss = torch.tensor(0.0, device=stop_logit.device)
    for logits, targets in zip(factor_logits, factor_targets):
        loss = loss + ce(logits, targets)
    loss = loss + bce(stop_logit.squeeze(-1), stop_target)
    return loss


# ── Evaluation ──────────────────────────────────────────────────────────────

def _greedy_match(pred_results, gt_indices, codebook_names):
    """Greedy matching of predicted objects to ground truth.

    Returns (correct_objects, total_objects, per_property_correct dict).
    """
    correct_objects = 0
    total_objects = 0
    correct_properties = {name: 0 for name in codebook_names}

    matched_gt = set()
    for pred_indices, conf in pred_results:
        best_match = -1
        best_score = -1
        for gi, gt in enumerate(gt_indices):
            if gi in matched_gt:
                continue
            score = sum(1 for p, g in zip(pred_indices, gt) if p == g)
            if score > best_score:
                best_score = score
                best_match = gi

        if best_match >= 0:
            matched_gt.add(best_match)
            gt = gt_indices[best_match]
            total_objects += 1

            all_correct = True
            for name, p, g in zip(codebook_names, pred_indices, gt):
                if p == g:
                    correct_properties[name] += 1
                else:
                    all_correct = False
            if all_correct:
                correct_objects += 1

    return correct_objects, total_objects, correct_properties


def evaluate_resonator(
    model: SceneEncoder,
    dataset,
    codebooks: SceneCodebooks,
    resonator: ResonatorNetwork,
    device: torch.device,
    n_samples: int = 100,
) -> dict:
    """Evaluate factorization accuracy using the resonator."""
    model.eval()
    correct_objects = 0
    total_objects = 0
    correct_properties = {name: 0 for name in codebooks.codebook_names()}

    sentinel_pairs = codebooks.discrete_sentinel_indices()

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            img, target, scene = dataset[i]
            img = img.unsqueeze(0).to(device)

            pred = model(img).squeeze(0).cpu()

            results = resonator.factorize_scene(
                pred, max_objects=len(scene), sentinel_pairs=sentinel_pairs,
            )

            gt_indices = [codebooks.object_to_indices(obj) for obj in scene]

            co, to, cp = _greedy_match(
                results, gt_indices, codebooks.codebook_names()
            )
            correct_objects += co
            total_objects += to
            for k in cp:
                correct_properties[k] += cp[k]

    if total_objects == 0:
        return {"object_acc": 0.0, "per_property": {k: 0.0 for k in correct_properties}}

    return {
        "object_acc": correct_objects / total_objects,
        "per_property": {
            k: v / total_objects for k, v in correct_properties.items()
        },
        "total_objects": total_objects,
    }


def evaluate_factorizer(
    model: SceneEncoder,
    factorizer: AmortizedFactorizer,
    dataset,
    codebooks: SceneCodebooks,
    device: torch.device,
    n_samples: int = 100,
) -> dict:
    """Evaluate factorization accuracy using the amortized factorizer."""
    model.eval()
    factorizer.eval()
    correct_objects = 0
    total_objects = 0
    correct_properties = {name: 0 for name in codebooks.codebook_names()}

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            img, target, scene = dataset[i]
            img = img.unsqueeze(0).to(device)

            pred = model(img).squeeze(0)

            results = factorizer.factorize_scene(
                pred, max_objects=len(scene),
            )

            gt_indices = [codebooks.object_to_indices(obj) for obj in scene]

            co, to, cp = _greedy_match(
                results, gt_indices, codebooks.codebook_names()
            )
            correct_objects += co
            total_objects += to
            for k in cp:
                correct_properties[k] += cp[k]

    if total_objects == 0:
        return {"object_acc": 0.0, "per_property": {k: 0.0 for k in correct_properties}}

    return {
        "object_acc": correct_objects / total_objects,
        "per_property": {
            k: v / total_objects for k, v in correct_properties.items()
        },
        "total_objects": total_objects,
    }


# ── Dataset / codebook builders ────────────────────────────────────────────

def build_codebooks(args) -> SceneCodebooks:
    """Create codebooks for the selected dataset."""
    if args.dataset == "toy":
        return SceneCodebooks.toy(
            d=args.dim,
            n_pos_levels=args.n_pos_levels,
            n_size_levels=args.n_size_levels,
        )
    elif args.dataset == "clevr":
        return SceneCodebooks.clevr(
            d=args.dim,
            n_pos_levels=args.n_pos_levels,
            n_size_levels=args.n_size_levels,
        )
    elif args.dataset == "dsprites":
        return SceneCodebooks.dsprites(
            d=args.dim,
            n_pos_levels=args.n_pos_levels,
            n_size_levels=args.n_size_levels,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def build_datasets(args, codebooks: SceneCodebooks):
    """Create train and val datasets for the selected dataset."""
    if args.dataset == "toy":
        from data.toy_scenes import ToySceneDataset
        train_dataset = ToySceneDataset(
            n_scenes=args.n_train,
            codebooks=codebooks,
            max_objects=args.max_objects,
            seed=0,
        )
        val_dataset = ToySceneDataset(
            n_scenes=args.n_val,
            codebooks=codebooks,
            max_objects=args.max_objects,
            seed=9999,
        )
    elif args.dataset == "clevr":
        from data.clevr_dataset import CLEVRDataset
        if not args.data_dir:
            raise ValueError("--data-dir is required for CLEVR dataset")
        train_dataset = CLEVRDataset(
            data_dir=args.data_dir,
            codebooks=codebooks,
            split="train",
            max_objects=args.max_objects,
            max_scenes=args.n_train,
        )
        val_dataset = CLEVRDataset(
            data_dir=args.data_dir,
            codebooks=codebooks,
            split="val",
            max_objects=args.max_objects,
            max_scenes=args.n_val,
        )
    elif args.dataset == "dsprites":
        from data.dsprites_dataset import MultiDSpritesDataset
        data_dir = args.data_dir or "."
        train_dataset = MultiDSpritesDataset(
            data_dir=data_dir,
            codebooks=codebooks,
            n_scenes=args.n_train,
            max_objects=args.max_objects,
            seed=0,
        )
        val_dataset = MultiDSpritesDataset(
            data_dir=data_dir,
            codebooks=codebooks,
            n_scenes=args.n_val,
            max_objects=args.max_objects,
            seed=9999,
        )
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return train_dataset, val_dataset


# ── Training loops ──────────────────────────────────────────────────────────

def train_encoder(args):
    """Phase 1: Train encoder with cosine loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Dataset: {args.dataset}")

    codebooks = build_codebooks(args)

    print("Loading/generating scenes...")
    train_dataset, val_dataset = build_datasets(args, codebooks)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    model = SceneEncoder(
        d=args.dim, backbone=args.backbone, freeze_backbone=args.freeze_backbone,
    ).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    resonator = ResonatorNetwork(codebooks.all_codebooks(), max_iters=200)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Backbone: {args.backbone} | Params: {trainable:,} trainable / {total:,} total")
    print(f"FHRR dim: {args.dim}, Max objects: {args.max_objects}")
    print(f"Factors: {codebooks.codebook_names()}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print()

    best_obj_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets, scenes in pbar:
            images = images.to(device)
            targets = targets.to(device)

            pred = model(images)
            loss = fhrr_cosine_loss(pred, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.shape[0]
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_dataset)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = evaluate_resonator(
                model, val_dataset, codebooks, resonator, device,
                n_samples=args.n_eval_samples,
            )
            print(
                f"  Epoch {epoch+1}: loss={avg_loss:.4f} | "
                f"obj_acc={metrics['object_acc']:.3f} | "
                + " ".join(
                    f"{k}={v:.3f}" for k, v in metrics["per_property"].items()
                )
            )
            if metrics["object_acc"] > best_obj_acc:
                best_obj_acc = metrics["object_acc"]
                torch.save(model.state_dict(), "best_model.pt")
                print(f"  ** New best: {best_obj_acc:.3f}")
        else:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    print(f"\nBest object accuracy: {best_obj_acc:.3f}")


def train_factorizer(args):
    """Phase 2: Train factorizer with frozen encoder."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: factorizer (Phase 2)")
    print(f"Dataset: {args.dataset}")

    codebooks = build_codebooks(args)

    print("Loading/generating scenes...")
    train_dataset, val_dataset = build_datasets(args, codebooks)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Load pretrained encoder (frozen)
    model = SceneEncoder(
        d=args.dim, backbone=args.backbone, freeze_backbone=True,
    ).to(device)
    if args.encoder_checkpoint:
        model.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device))
        print(f"Loaded encoder from {args.encoder_checkpoint}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Build factorizer
    factorizer = AmortizedFactorizer(
        codebooks,
        hidden_dim=args.factorizer_hidden,
        n_hidden_layers=args.factorizer_layers,
        dropout=args.factorizer_dropout,
    ).to(device)

    optimizer = torch.optim.Adam(factorizer.parameters(), lr=args.factorizer_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Precompute stop vector
    stop_vec = codebooks.stop_vector.to(device)

    trainable = sum(p.numel() for p in factorizer.parameters())
    print(f"Factorizer params: {trainable:,}")
    print(f"Codebook sizes: {factorizer.codebook_sizes} (total_sim_dim={factorizer.total_sim_dim})")
    print(f"FHRR dim: {args.dim}, Max objects: {args.max_objects}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print()

    best_obj_acc = 0.0

    for epoch in range(args.epochs):
        factorizer.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets, scenes in pbar:
            images = images.to(device)
            targets = targets.to(device)

            # Get encoder predictions (frozen)
            with torch.no_grad():
                pred = model(images)

            # Factorizer forward on encoder predictions
            factor_logits, stop_logit = factorizer(pred)
            factor_targets, stop_targets = compute_factor_targets(
                scenes, codebooks, device
            )
            loss = factorizer_loss(
                factor_logits, stop_logit, factor_targets, stop_targets
            )

            # STOP training: only train the stop head (BCE), not factor MLP
            B = images.shape[0]
            stop_batch = stop_vec.unsqueeze(0).expand(B, -1)
            _, stop_stop_logit = factorizer(stop_batch)
            stop_loss = nn.BCEWithLogitsLoss()(
                stop_stop_logit.squeeze(-1), torch.ones(B, device=device)
            )

            total_batch_loss = loss + stop_loss

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item() * B
            pbar.set_postfix(loss=f"{total_batch_loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_dataset)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = evaluate_factorizer(
                model, factorizer, val_dataset, codebooks, device,
                n_samples=args.n_eval_samples,
            )
            print(
                f"  Epoch {epoch+1}: loss={avg_loss:.4f} | "
                f"obj_acc={metrics['object_acc']:.3f} | "
                + " ".join(
                    f"{k}={v:.3f}" for k, v in metrics["per_property"].items()
                )
            )
            if metrics["object_acc"] > best_obj_acc:
                best_obj_acc = metrics["object_acc"]
                torch.save(factorizer.state_dict(), "best_factorizer.pt")
                print(f"  ** New best: {best_obj_acc:.3f}")
        else:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    print(f"\nBest object accuracy: {best_obj_acc:.3f}")


def train_joint(args):
    """Phase 3: Joint fine-tuning of encoder + factorizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: joint (Phase 3)")
    print(f"Dataset: {args.dataset}")

    codebooks = build_codebooks(args)

    print("Loading/generating scenes...")
    train_dataset, val_dataset = build_datasets(args, codebooks)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Load encoder (trainable)
    model = SceneEncoder(
        d=args.dim, backbone=args.backbone, freeze_backbone=args.freeze_backbone,
    ).to(device)
    if args.encoder_checkpoint:
        model.load_state_dict(torch.load(args.encoder_checkpoint, map_location=device))
        print(f"Loaded encoder from {args.encoder_checkpoint}")

    # Load factorizer
    factorizer = AmortizedFactorizer(
        codebooks,
        hidden_dim=args.factorizer_hidden,
        n_hidden_layers=args.factorizer_layers,
        dropout=args.factorizer_dropout,
    ).to(device)
    if args.factorizer_checkpoint:
        factorizer.load_state_dict(
            torch.load(args.factorizer_checkpoint, map_location=device)
        )
        print(f"Loaded factorizer from {args.factorizer_checkpoint}")

    # Joint optimizer
    all_params = list(filter(lambda p: p.requires_grad, model.parameters())) + \
                 list(factorizer.parameters())
    optimizer = torch.optim.Adam(all_params, lr=args.factorizer_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    stop_vec = codebooks.stop_vector.to(device)

    enc_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    fac_params = sum(p.numel() for p in factorizer.parameters())
    print(f"Encoder params: {enc_params:,} trainable | Factorizer params: {fac_params:,}")
    print(f"FHRR dim: {args.dim}, Max objects: {args.max_objects}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    print()

    best_obj_acc = 0.0

    for epoch in range(args.epochs):
        model.train()
        factorizer.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for images, targets, scenes in pbar:
            images = images.to(device)
            targets = targets.to(device)

            # Encoder forward (gradients flow)
            pred = model(images)

            # Cosine loss on encoder
            cos_loss = fhrr_cosine_loss(pred, targets)

            # Factorizer loss (detach encoder output — cosine loss supervises
            # encoder, factorizer learns to decode without destabilizing it)
            factor_logits, stop_logit = factorizer(pred.detach())
            factor_targets, stop_targets = compute_factor_targets(
                scenes, codebooks, device
            )
            fac_loss = factorizer_loss(
                factor_logits, stop_logit, factor_targets, stop_targets
            )

            # STOP training: only train the stop head (BCE), not factor MLP
            B = images.shape[0]
            stop_batch = stop_vec.unsqueeze(0).expand(B, -1)
            _, stop_stop_logit = factorizer(stop_batch)
            stop_loss = nn.BCEWithLogitsLoss()(
                stop_stop_logit.squeeze(-1), torch.ones(B, device=device)
            )

            loss = cos_loss + fac_loss + stop_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * B
            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                cos=f"{cos_loss.item():.4f}",
                ce=f"{fac_loss.item():.4f}",
            )

        scheduler.step()
        avg_loss = total_loss / len(train_dataset)

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = evaluate_factorizer(
                model, factorizer, val_dataset, codebooks, device,
                n_samples=args.n_eval_samples,
            )
            print(
                f"  Epoch {epoch+1}: loss={avg_loss:.4f} | "
                f"obj_acc={metrics['object_acc']:.3f} | "
                + " ".join(
                    f"{k}={v:.3f}" for k, v in metrics["per_property"].items()
                )
            )
            if metrics["object_acc"] > best_obj_acc:
                best_obj_acc = metrics["object_acc"]
                torch.save(model.state_dict(), "best_model_joint.pt")
                torch.save(factorizer.state_dict(), "best_factorizer_joint.pt")
                print(f"  ** New best: {best_obj_acc:.3f}")
        else:
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}")

    print(f"\nBest object accuracy: {best_obj_acc:.3f}")


# ── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument("--mode", type=str, default="encoder",
                        choices=["encoder", "factorizer", "joint"],
                        help="Training mode: encoder (Phase 1), factorizer (Phase 2), joint (Phase 3)")

    # Dataset
    parser.add_argument("--dataset", type=str, default="toy",
                        choices=["toy", "clevr", "dsprites"],
                        help="Dataset to train on")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (required for CLEVR, optional for dSprites)")

    # Encoder
    parser.add_argument("--backbone", type=str, default="simple",
                        choices=["simple", "resnet"],
                        help="CNN backbone: simple (lightweight) or resnet (pretrained ResNet-34)")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Freeze backbone weights, only train MLP head")
    parser.add_argument("--encoder-checkpoint", type=str, default=None,
                        help="Path to pretrained encoder checkpoint (for factorizer/joint modes)")

    # Factorizer
    parser.add_argument("--factorizer-hidden", type=int, default=256,
                        help="Factorizer MLP hidden dimension")
    parser.add_argument("--factorizer-layers", type=int, default=2,
                        help="Number of hidden layers in factorizer MLP")
    parser.add_argument("--factorizer-dropout", type=float, default=0.1,
                        help="Factorizer dropout rate")
    parser.add_argument("--factorizer-lr", type=float, default=1e-3,
                        help="Factorizer learning rate")
    parser.add_argument("--factorizer-checkpoint", type=str, default=None,
                        help="Path to pretrained factorizer checkpoint (for joint mode)")

    # FHRR / codebook
    parser.add_argument("--dim", type=int, default=1024, help="FHRR dimension")
    parser.add_argument("--n-pos-levels", type=int, default=32)
    parser.add_argument("--n-size-levels", type=int, default=16)

    # Data
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--max-objects", type=int, default=1,
                        help="Start with 1 for Phase 1")

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--n-eval-samples", type=int, default=200)

    args = parser.parse_args()

    if args.mode == "encoder":
        train_encoder(args)
    elif args.mode == "factorizer":
        train_factorizer(args)
    elif args.mode == "joint":
        train_joint(args)
