"""Amortized Factorizer: learned replacement for the resonator.

Takes an FHRR vector, computes similarities to each codebook,
and uses a cross-factor MLP to predict per-factor codebook indices
in a single forward pass. Includes a STOP head for peeling termination.
"""

import torch
import torch.nn as nn

from vsa.codebooks import SceneCodebooks
from vsa.fhrr import similarity, bind


class AmortizedFactorizer(nn.Module):
    """MLP that maps FHRR codebook similarities to per-factor logits."""

    def __init__(
        self,
        codebooks: SceneCodebooks,
        hidden_dim: int = 256,
        n_hidden_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.codebooks = codebooks

        # Register codebook tensors as buffers (not parameters)
        cb_list = codebooks.all_codebooks()
        self.n_factors = len(cb_list)
        self.codebook_sizes = [cb.shape[0] for cb in cb_list]
        self.total_sim_dim = sum(self.codebook_sizes)

        for i, cb in enumerate(cb_list):
            # Store real and imag parts since buffers must be real
            self.register_buffer(f"cb_real_{i}", cb.real.clone())
            self.register_buffer(f"cb_imag_{i}", cb.imag.clone())

        # Cross-factor MLP
        layers = []
        in_dim = self.total_sim_dim
        for _ in range(n_hidden_layers):
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.total_sim_dim))
        self.mlp = nn.Sequential(*layers)

        # STOP head: binary classifier
        self.stop_head = nn.Sequential(
            nn.Linear(self.total_sim_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _get_codebook(self, i: int) -> torch.Tensor:
        """Reconstruct complex codebook from stored real/imag buffers."""
        real = getattr(self, f"cb_real_{i}")
        imag = getattr(self, f"cb_imag_{i}")
        return torch.complex(real, imag)

    def compute_similarities(self, z: torch.Tensor) -> torch.Tensor:
        """Compute similarity of z against each codebook.

        Args:
            z: (B, d) complex FHRR vectors

        Returns:
            (B, total_sim_dim) real-valued similarities
        """
        sims = []
        for i in range(self.n_factors):
            cb = self._get_codebook(i)
            # similarity: (B, d) vs (Nk, d) → (B, Nk)
            sims.append(similarity(z, cb))
        return torch.cat(sims, dim=-1)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Forward pass: FHRR vector → per-factor logits + stop logit.

        Args:
            z: (B, d) complex FHRR vectors

        Returns:
            factor_logits: list of K tensors, each (B, Nk)
            stop_logit: (B, 1)
        """
        sims = self.compute_similarities(z)  # (B, total_sim_dim)
        refined = self.mlp(sims)  # (B, total_sim_dim)
        stop_logit = self.stop_head(sims)  # (B, 1)

        # Split refined output into per-factor logits
        factor_logits = []
        offset = 0
        for size in self.codebook_sizes:
            factor_logits.append(refined[:, offset:offset + size])
            offset += size

        return factor_logits, stop_logit

    @torch.no_grad()
    def predict_indices(self, z: torch.Tensor) -> tuple[list[int], float]:
        """Inference helper: predict codebook indices for a single vector.

        Args:
            z: (d,) complex FHRR vector

        Returns:
            (indices, stop_prob) — list of K ints, and stop probability
        """
        self.eval()
        z_batch = z.unsqueeze(0)  # (1, d)
        factor_logits, stop_logit = self.forward(z_batch)

        indices = [logits.argmax(dim=-1).item() for logits in factor_logits]
        stop_prob = torch.sigmoid(stop_logit).item()
        return indices, stop_prob

    @torch.no_grad()
    def factorize_scene(
        self,
        s: torch.Tensor,
        max_objects: int = 5,
        stop_threshold: float = 0.5,
    ) -> list[tuple[list[int], float]]:
        """Drop-in replacement for ResonatorNetwork.factorize_scene.

        Uses sequential peeling with the learned STOP head.

        Args:
            s: scene FHRR vector (d,) complex
            max_objects: maximum objects to extract
            stop_threshold: STOP probability threshold

        Returns:
            List of (codebook_indices, confidence) per extracted object.
        """
        self.eval()
        residual = s.clone()
        results = []

        for _ in range(max_objects):
            indices, stop_prob = self.predict_indices(residual)

            if stop_prob > stop_threshold:
                break

            # Confidence: similarity between residual and reconstructed binding
            parts = [self._get_codebook(i)[idx] for i, idx in enumerate(indices)]
            reconstructed = bind(*parts)
            conf = similarity(
                residual.unsqueeze(0), reconstructed.unsqueeze(0)
            ).item()

            results.append((indices, conf))

            # Peel: subtract reconstructed object
            residual = residual - reconstructed

        return results
