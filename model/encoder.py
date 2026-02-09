"""Scene encoder: CNN → MLP → FHRR vector.

Maps a 2D image to a complex-valued vector on the unit circle,
trained so that the output approximates the ground-truth FHRR
scene encoding (superposition of object bindings).
"""

import torch
import torch.nn as nn


class SceneEncoder(nn.Module):
    """Encodes an image into an FHRR vector."""

    def __init__(self, d: int, img_size: int = 64):
        """
        Args:
            d: FHRR vector dimensionality
            img_size: input image size (square)
        """
        super().__init__()
        self.d = d

        # Simple CNN backbone
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # 64→32
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 32→16
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), # 16→8
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), # 8→4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),                     # 4→1
            nn.Flatten(),                                 # → (B, 256)
        )

        # MLP: maps CNN features to complex FHRR vector
        # Output 2*d real values, interpreted as (real, imag) pairs
        self.mlp = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 2 * d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: input images, shape (B, 3, H, W)

        Returns:
            FHRR vectors on the unit circle, shape (B, d), complex64
        """
        features = self.cnn(x)
        out = self.mlp(features)  # (B, 2*d)

        # Split into real and imaginary, form complex, project to unit circle
        real = out[:, :self.d]
        imag = out[:, self.d:]
        z = torch.complex(real, imag)

        # Normalize to unit circle
        z = z / z.abs().clamp(min=1e-8)

        return z
