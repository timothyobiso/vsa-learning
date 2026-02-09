"""Scene encoder: CNN → MLP → FHRR vector.

Maps a 2D image to a complex-valued vector on the unit circle,
trained so that the output approximates the ground-truth FHRR
scene encoding (superposition of object bindings).

Supports two backbone modes:
  - "simple": lightweight 4-layer CNN (good for toy/dSprites)
  - "resnet": pretrained ResNet-34 (better for CLEVR / complex scenes)
"""

import torch
import torch.nn as nn
from torchvision import models


class SceneEncoder(nn.Module):
    """Encodes an image into an FHRR vector."""

    def __init__(self, d: int, backbone: str = "simple", freeze_backbone: bool = False):
        """
        Args:
            d: FHRR vector dimensionality
            backbone: "simple" for lightweight CNN, "resnet" for pretrained ResNet-34
            freeze_backbone: if True, freeze backbone weights (only train MLP head)
        """
        super().__init__()
        self.d = d
        self.backbone_name = backbone

        if backbone == "simple":
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
            feat_dim = 256
        elif backbone == "resnet":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            # Remove the final FC layer, keep everything up to avgpool
            self.cnn = nn.Sequential(*list(resnet.children())[:-1], nn.Flatten())
            feat_dim = 512

            if freeze_backbone:
                for param in self.cnn.parameters():
                    param.requires_grad = False
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # MLP: maps CNN features to complex FHRR vector
        # Output 2*d real values, interpreted as (real, imag) pairs
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 512),
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
