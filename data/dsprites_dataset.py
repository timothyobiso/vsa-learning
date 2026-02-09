"""Multi-dSprites dataset loader.

Generates multi-object scenes by compositing sprites from the base dSprites
dataset onto an RGB canvas with random colors.

Properties per object:
  - shape: "ellipse" / "heart" / "square" → discrete
  - R, G, B: color channels in [0, 1] → FPE
  - x, y: position in [0, 1] → FPE
  - scale: size in [0, 1] → FPE

Download dSprites:
  https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz

Place the .npz file at <data_dir>/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
or pass the path directly.
"""

import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from vsa.codebooks import DSPRITES_SHAPES, SceneCodebooks


IMG_SIZE = 64

# dSprites shape indices: 0=square, 1=ellipse, 2=heart (in the .npz)
# We remap to our naming convention
_DSPRITES_SHAPE_NAMES = {0: "square", 1: "ellipse", 2: "heart"}

# dSprites latent factor indices in the .npz:
# 0: color (always 1, white), 1: shape, 2: scale, 3: orientation, 4: posX, 5: posY
# latents_values ranges: scale=[0.5..1.0], orientation=[0..2pi], posX=[0..1], posY=[0..1]


def load_dsprites(data_dir: str) -> dict:
    """Load the base dSprites .npz file.

    Returns dict with 'imgs' (737280, 64, 64) uint8 and 'latents_values' (737280, 6).
    """
    data_path = Path(data_dir)
    npz_file = data_path / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
    data = np.load(npz_file, allow_pickle=True)
    return {
        "imgs": data["imgs"],  # (N, 64, 64) binary
        "latents_values": data["latents_values"],  # (N, 6) float
        "latents_classes": data["latents_classes"],  # (N, 6) int
    }


def _composite_sprite(
    canvas: np.ndarray,
    sprite_img: np.ndarray,
    color_rgb: tuple[float, float, float],
):
    """Composite a binary sprite onto an RGB canvas with given color.

    Modifies canvas in-place. sprite_img is (64, 64) binary.
    """
    mask = sprite_img.astype(bool)
    for c in range(3):
        canvas[c][mask] = color_rgb[c]


class MultiDSpritesDataset(Dataset):
    """PyTorch dataset of multi-object dSprites scenes with FHRR targets.

    Generates scenes on-the-fly by compositing random sprites onto a canvas.

    Args:
        data_dir: directory containing dsprites .npz file
        codebooks: SceneCodebooks instance (use SceneCodebooks.dsprites(d))
        n_scenes: number of scenes to generate
        max_objects: maximum objects per scene
        seed: random seed for reproducibility
    """

    def __init__(
        self,
        data_dir: str,
        codebooks: SceneCodebooks,
        n_scenes: int = 5000,
        max_objects: int = 2,
        seed: int = 0,
    ):
        self.codebooks = codebooks

        # Load base dSprites data
        dsprites = load_dsprites(data_dir)
        imgs = dsprites["imgs"]
        latents_values = dsprites["latents_values"]
        latents_classes = dsprites["latents_classes"]

        # Group sprite indices by shape for easy sampling
        # shape class: 0=square, 1=ellipse, 2=heart
        self._sprites_by_shape = {}
        for shape_cls in range(3):
            mask = latents_classes[:, 1] == shape_cls
            self._sprites_by_shape[shape_cls] = np.where(mask)[0]

        self.images = []
        self.scenes = []
        self.targets = []

        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)

        for _ in range(n_scenes):
            n_obj = rng.randint(1, max_objects)

            # Start with black canvas: (3, 64, 64) float [0, 1]
            canvas = np.zeros((3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
            obj_dicts = []

            for _ in range(n_obj):
                # Pick a random shape
                shape_cls = rng.randint(0, 2)
                shape_name = _DSPRITES_SHAPE_NAMES[shape_cls]

                # Pick a random sprite of this shape
                candidates = self._sprites_by_shape[shape_cls]
                sprite_idx = candidates[np_rng.randint(len(candidates))]

                sprite_img = imgs[sprite_idx]
                lat_vals = latents_values[sprite_idx]

                # Extract position and scale from latent values
                # lat_vals: [color, shape, scale, orientation, posX, posY]
                scale_raw = lat_vals[2]  # in [0.5, 1.0]
                pos_x = lat_vals[4]      # in [0, 1]
                pos_y = lat_vals[5]      # in [0, 1]

                # Normalize scale from [0.5, 1.0] to [0, 1]
                scale_norm = (scale_raw - 0.5) / 0.5

                # Random color
                r_val = rng.random()
                g_val = rng.random()
                b_val = rng.random()

                # Composite onto canvas
                _composite_sprite(canvas, sprite_img, (r_val, g_val, b_val))

                obj_dicts.append({
                    "shape": shape_name,
                    "R": r_val,
                    "G": g_val,
                    "B": b_val,
                    "x": pos_x,
                    "y": pos_y,
                    "scale": scale_norm,
                })

            img_tensor = torch.from_numpy(canvas.copy())
            target = codebooks.encode_scene(obj_dicts)

            self.images.append(img_tensor)
            self.scenes.append(obj_dicts)
            self.targets.append(target)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        return self.images[idx], self.targets[idx], self.scenes[idx]
