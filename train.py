"""Training loop for the scene encoder.

Loss: cosine similarity between predicted FHRR vector and ground-truth
scene encoding (superposition of object bindings).
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from vsa.codebooks import SceneCodebooks
from vsa.resonator import ResonatorNetwork
from data.toy_scenes import ToySceneDataset
from model.encoder import SceneEncoder


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


def evaluate_resonator(
    model: SceneEncoder,
    dataset: ToySceneDataset,
    codebooks: SceneCodebooks,
    resonator: ResonatorNetwork,
    device: torch.device,
    n_samples: int = 100,
) -> dict:
    """Evaluate factorization accuracy on a subset of the dataset."""
    model.eval()
    correct_objects = 0
    total_objects = 0
    correct_properties = {name: 0 for name in codebooks.codebook_names()}

    with torch.no_grad():
        for i in range(min(n_samples, len(dataset))):
            img, target, scene = dataset[i]
            img = img.unsqueeze(0).to(device)

            pred = model(img).squeeze(0).cpu()

            # Run resonator on predicted vector
            results = resonator.factorize_scene(pred, max_objects=len(scene))

            # Match predicted objects to ground-truth
            gt_indices = []
            for obj in scene:
                gt_idx = [
                    codebooks.shape_to_idx[obj["shape"]],
                    codebooks.color_to_idx[obj["color"]],
                    _nearest_fpe_idx(obj["x"], codebooks.n_pos_levels),
                    _nearest_fpe_idx(obj["y"], codebooks.n_pos_levels),
                    _nearest_fpe_idx(obj["size"], codebooks.n_size_levels),
                ]
                gt_indices.append(gt_idx)

            # Greedy matching: for each predicted object, find best GT match
            matched_gt = set()
            for pred_indices, conf in results:
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
                    for pi, (name, p, g) in enumerate(
                        zip(codebooks.codebook_names(), pred_indices, gt)
                    ):
                        if p == g:
                            correct_properties[name] += 1
                        else:
                            all_correct = False
                    if all_correct:
                        correct_objects += 1

    if total_objects == 0:
        return {"object_acc": 0.0, "per_property": {k: 0.0 for k in correct_properties}}

    return {
        "object_acc": correct_objects / total_objects,
        "per_property": {
            k: v / total_objects for k, v in correct_properties.items()
        },
        "total_objects": total_objects,
    }


def _nearest_fpe_idx(value: float, n_levels: int) -> int:
    """Map a [0,1] value to the nearest FPE codebook index."""
    return round(value * (n_levels - 1))


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create codebooks
    codebooks = SceneCodebooks(
        d=args.dim,
        n_pos_levels=args.n_pos_levels,
        n_size_levels=args.n_size_levels,
    )

    # Create datasets
    print("Generating training scenes...")
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

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Model
    model = SceneEncoder(d=args.dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # Resonator for evaluation
    resonator = ResonatorNetwork(codebooks.all_codebooks(), max_iters=200)

    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"FHRR dim: {args.dim}, Max objects: {args.max_objects}")
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

        # Evaluate every few epochs
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=1024, help="FHRR dimension")
    parser.add_argument("--n-train", type=int, default=5000)
    parser.add_argument("--n-val", type=int, default=500)
    parser.add_argument("--max-objects", type=int, default=1,
                        help="Start with 1 for Phase 1")
    parser.add_argument("--n-pos-levels", type=int, default=32)
    parser.add_argument("--n-size-levels", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--n-eval-samples", type=int, default=200)
    args = parser.parse_args()
    train(args)
