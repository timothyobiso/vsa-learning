"""Scene codebook management.

Creates and stores the fixed codebooks for scene properties:
- Shape: discrete (circle, square, triangle)
- Color: discrete (red, green, blue)
- X position: continuous via FPE
- Y position: continuous via FPE
- Size: continuous via FPE
"""

import torch
from .fhrr import random_vectors
from .fpe import fpe_codebook


SHAPES = ["circle", "square", "triangle", "none"]
COLORS = ["red", "green", "blue", "none"]


class SceneCodebooks:
    """Fixed codebooks for scene factorization."""

    def __init__(self, d: int, n_pos_levels: int = 32, n_size_levels: int = 16, seed: int = 42):
        self.d = d
        self.n_pos_levels = n_pos_levels
        self.n_size_levels = n_size_levels

        gen = torch.Generator().manual_seed(seed)

        # Discrete codebooks: one random vector per item
        self.shape_vectors = random_vectors(len(SHAPES), d, generator=gen)
        self.color_vectors = random_vectors(len(COLORS), d, generator=gen)

        # FPE base vectors for continuous properties
        self.x_base = random_vectors(1, d, generator=gen).squeeze(0)
        self.y_base = random_vectors(1, d, generator=gen).squeeze(0)
        self.size_base = random_vectors(1, d, generator=gen).squeeze(0)

        # FPE codebooks
        self.x_codebook = fpe_codebook(self.x_base, n_pos_levels)
        self.y_codebook = fpe_codebook(self.y_base, n_pos_levels)
        self.size_codebook = fpe_codebook(self.size_base, n_size_levels)

        # Name-to-index mappings
        self.shape_to_idx = {s: i for i, s in enumerate(SHAPES)}
        self.color_to_idx = {c: i for i, c in enumerate(COLORS)}

    def to(self, device: torch.device) -> "SceneCodebooks":
        """Move all codebooks to a device."""
        self.shape_vectors = self.shape_vectors.to(device)
        self.color_vectors = self.color_vectors.to(device)
        self.x_base = self.x_base.to(device)
        self.y_base = self.y_base.to(device)
        self.size_base = self.size_base.to(device)
        self.x_codebook = self.x_codebook.to(device)
        self.y_codebook = self.y_codebook.to(device)
        self.size_codebook = self.size_codebook.to(device)
        return self

    def all_codebooks(self) -> list[torch.Tensor]:
        """Return list of all codebooks for the resonator."""
        return [
            self.shape_vectors,
            self.color_vectors,
            self.x_codebook,
            self.y_codebook,
            self.size_codebook,
        ]

    def codebook_names(self) -> list[str]:
        return ["shape", "color", "x", "y", "size"]

    @property
    def stop_vector(self) -> torch.Tensor:
        """STOP object: bind(none_shape, none_color, x=0, y=0, size=0)."""
        from .fhrr import bind

        s = self.shape_vectors[self.shape_to_idx["none"]]
        c = self.color_vectors[self.color_to_idx["none"]]
        xv = self.x_codebook[0]
        yv = self.y_codebook[0]
        sv = self.size_codebook[0]
        return bind(s, c, xv, yv, sv)

    def encode_object(
        self, shape: str, color: str, x: float, y: float, size: float
    ) -> torch.Tensor:
        """Encode a single object as a bound FHRR vector.

        Args:
            shape: one of SHAPES
            color: one of COLORS
            x, y: position in [0, 1]
            size: size in [0, 1]
        """
        from .fhrr import bind
        from .fpe import fpe_encode

        s = self.shape_vectors[self.shape_to_idx[shape]]
        c = self.color_vectors[self.color_to_idx[color]]
        xv = fpe_encode(self.x_base, x)
        yv = fpe_encode(self.y_base, y)
        sv = fpe_encode(self.size_base, size)
        return bind(s, c, xv, yv, sv)

    def encode_scene(self, objects: list[dict]) -> torch.Tensor:
        """Encode a full scene as a bundled superposition of object bindings.

        Args:
            objects: list of dicts with keys: shape, color, x, y, size
        """
        from .fhrr import bundle

        bindings = [self.encode_object(**obj) for obj in objects]
        scene = bundle(*bindings)
        # Add STOP vector at reduced weight so real objects are peeled first.
        # Weight 0.5 means STOP is always weaker than any single real object
        # (each real object has weight 1 in the sum).
        scene = scene + 0.5 * self.stop_vector
        return scene
