"""Toy 2D scene generator.

Generates simple images with 1-N colored geometric shapes at various positions
and sizes. Each scene comes with ground-truth object descriptions.

Phase 1: 1-2 objects, 3 shapes (circle, square, triangle), 3 colors (red, green, blue),
continuous position and size.
"""

import math
import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

from vsa.codebooks import SHAPES, COLORS, SceneCodebooks


IMG_SIZE = 64

# Exclude "none" sentinel from random scene generation
REAL_SHAPES = [s for s in SHAPES if s != "none"]
REAL_COLORS = [c for c in COLORS if c != "none"]

COLOR_RGB = {
    "red": (220, 50, 50),
    "green": (50, 200, 50),
    "blue": (50, 50, 220),
}


@dataclass
class SceneObject:
    shape: str
    color: str
    x: float  # center x in [0, 1]
    y: float  # center y in [0, 1]
    size: float  # relative size in [0.15, 0.85] → mapped to [0, 1] for FPE

    def to_dict(self) -> dict:
        return {
            "shape": self.shape,
            "color": self.color,
            "x": self.x,
            "y": self.y,
            "size": self.size,
        }


def draw_scene(objects: list[SceneObject], img_size: int = IMG_SIZE) -> Image.Image:
    """Render a scene as a PIL image."""
    img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    for obj in objects:
        cx = obj.x * img_size
        cy = obj.y * img_size
        # Map size [0, 1] → pixel radius [4, 20]
        r = 4 + obj.size * 16
        rgb = COLOR_RGB[obj.color]

        if obj.shape == "circle":
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=rgb)
        elif obj.shape == "square":
            draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=rgb)
        elif obj.shape == "triangle":
            pts = [
                (cx, cy - r),
                (cx - r * math.cos(math.radians(30)), cy + r * math.sin(math.radians(30))),
                (cx + r * math.cos(math.radians(30)), cy + r * math.sin(math.radians(30))),
            ]
            draw.polygon(pts, fill=rgb)

    return img


def generate_scene(
    n_objects: int,
    rng=None,
) -> tuple[list[SceneObject], Image.Image]:
    """Generate a random scene with n_objects.

    Positions are kept in [0.15, 0.85] to avoid clipping at edges.
    """
    if rng is None:
        rng = random.Random()

    objects = []
    for _ in range(n_objects):
        obj = SceneObject(
            shape=rng.choice(REAL_SHAPES),
            color=rng.choice(REAL_COLORS),
            x=rng.uniform(0.15, 0.85),
            y=rng.uniform(0.15, 0.85),
            size=rng.uniform(0.1, 0.9),
        )
        objects.append(obj)

    img = draw_scene(objects)
    return objects, img


class ToySceneDataset(Dataset):
    """PyTorch dataset of toy scenes with FHRR target vectors."""

    def __init__(
        self,
        n_scenes: int,
        codebooks: SceneCodebooks,
        max_objects: int = 2,
        seed: int = 0,
    ):
        self.codebooks = codebooks
        self.scenes = []
        self.images = []
        self.targets = []

        rng = random.Random(seed)

        for _ in range(n_scenes):
            n_obj = rng.randint(1, max_objects)
            objects, img = generate_scene(n_obj, rng)

            # Convert image to tensor: (3, H, W), float [0, 1]
            img_tensor = torch.tensor(
                list(img.getdata()), dtype=torch.float32
            ).reshape(IMG_SIZE, IMG_SIZE, 3).permute(2, 0, 1) / 255.0

            # Build target FHRR vector
            obj_dicts = [o.to_dict() for o in objects]
            target = codebooks.encode_scene(obj_dicts)

            self.scenes.append(obj_dicts)
            self.images.append(img_tensor)
            self.targets.append(target)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
        return self.images[idx], self.targets[idx], self.scenes[idx]
