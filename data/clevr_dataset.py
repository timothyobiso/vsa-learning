"""CLEVR dataset loader.

Loads CLEVR v1.0 images and scene annotations, encodes objects as FHRR targets.

Properties per object:
  - shape: "cube" / "sphere" / "cylinder" → discrete
  - color: 8 CLEVR colors → discrete
  - material: "metal" / "rubber" → discrete
  - x, y, z: 3D position normalized to [0,1] → FPE
  - size: "small" / "large" mapped to continuous [0,1] → FPE

Download CLEVR v1.0:
  https://cs.stanford.edu/people/jcjohns/clevr/
  Extract so that data_dir contains: images/, scenes/

Expected directory structure:
  <data_dir>/
    images/
      train/  CLEVR_train_000000.png ...
      val/    CLEVR_val_000000.png ...
    scenes/
      CLEVR_train_scenes.json
      CLEVR_val_scenes.json
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from vsa.codebooks import SceneCodebooks


IMG_SIZE = 64

# CLEVR coordinate ranges (approximate, from dataset)
CLEVR_X_RANGE = (-3.0, 3.0)
CLEVR_Y_RANGE = (-3.0, 3.0)
CLEVR_Z_RANGE = (0.0, 1.5)

# CLEVR size names → continuous values in [0, 1]
CLEVR_SIZE_MAP = {"small": 0.25, "large": 0.75}

_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),  # → (3, H, W) float [0, 1]
])


def _normalize(val: float, lo: float, hi: float) -> float:
    """Normalize a value from [lo, hi] to [0, 1], clamped."""
    return max(0.0, min(1.0, (val - lo) / (hi - lo)))


def _parse_clevr_object(obj: dict) -> dict:
    """Convert a CLEVR scene JSON object to a property dict for codebook encoding."""
    x = _normalize(obj["3d_coords"][0], *CLEVR_X_RANGE)
    y = _normalize(obj["3d_coords"][1], *CLEVR_Y_RANGE)
    z = _normalize(obj["3d_coords"][2], *CLEVR_Z_RANGE)
    size_val = CLEVR_SIZE_MAP[obj["size"]]

    # CLEVR uses "cube"/"sphere"/"cylinder" natively
    return {
        "shape": obj["shape"],
        "color": obj["color"],
        "material": obj["material"],
        "x": x,
        "y": y,
        "z": z,
        "size": size_val,
    }


class CLEVRDataset(Dataset):
    """PyTorch dataset for CLEVR with FHRR target vectors.

    Args:
        data_dir: path to CLEVR root directory
        codebooks: SceneCodebooks instance (use SceneCodebooks.clevr(d))
        split: "train" or "val"
        max_objects: maximum objects per scene (scenes with more are skipped)
        max_scenes: limit number of scenes loaded (None = all)
    """

    def __init__(
        self,
        data_dir: str,
        codebooks: SceneCodebooks,
        split: str = "train",
        max_objects: int = 5,
        max_scenes: int | None = None,
    ):
        self.codebooks = codebooks
        self.data_dir = Path(data_dir)
        self.split = split

        # Load scene annotations
        scene_file = self.data_dir / "scenes" / f"CLEVR_{split}_scenes.json"
        with open(scene_file) as f:
            all_scenes = json.load(f)["scenes"]

        self.image_paths = []
        self.scenes = []
        self.targets = []

        for scene_info in all_scenes:
            objects = scene_info["objects"]
            if len(objects) > max_objects or len(objects) == 0:
                continue

            obj_dicts = [_parse_clevr_object(obj) for obj in objects]

            # Build image path
            img_filename = scene_info["image_filename"]
            img_path = self.data_dir / "images" / split / img_filename

            target = codebooks.encode_scene(obj_dicts)

            self.image_paths.append(img_path)
            self.scenes.append(obj_dicts)
            self.targets.append(target)

            if max_scenes is not None and len(self.scenes) >= max_scenes:
                break

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        img = Image.open(self.image_paths[idx]).convert("RGB")
        img_tensor = _transform(img)
        return img_tensor, self.targets[idx], self.scenes[idx]
