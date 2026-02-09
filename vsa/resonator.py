"""Resonator Network for factorizing FHRR vectors.

Given a composed vector S = c1 ⊛ c2 ⊛ ... ⊛ ck (binding of factors),
the resonator iteratively estimates each factor by:
    1. Unbind S with current estimates of all OTHER factors
    2. Project the result onto the factor's codebook (nearest neighbor)
    3. Repeat until convergence

For scenes with multiple objects (superposition), we use sequential peeling:
    1. Run the resonator to find one object's factors
    2. Reconstruct that object's binding and subtract from S
    3. Repeat on the residual
"""

import torch
from .fhrr import bind, unbind, similarity, normalize, bundle


class ResonatorNetwork:
    """Iterative resonator for FHRR vector factorization."""

    def __init__(
        self,
        codebooks: list[torch.Tensor],
        max_iters: int = 100,
        convergence_threshold: float = 1e-6,
        n_restarts: int = 5,
    ):
        """
        Args:
            codebooks: list of K codebooks, each shape (Nk, d)
            max_iters: maximum resonator iterations per restart
            convergence_threshold: stop when estimates don't change
            n_restarts: number of random restarts (best result kept)
        """
        self.codebooks = codebooks
        self.max_iters = max_iters
        self.convergence_threshold = convergence_threshold
        self.n_restarts = n_restarts
        self.n_factors = len(codebooks)

    def _similarities(self, x: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        """Compute similarity of x against all items in codebook."""
        return similarity(x.unsqueeze(0), codebook).squeeze(0)

    def _project(self, x: torch.Tensor, codebook: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Project x onto the nearest vector in the codebook."""
        sims = self._similarities(x, codebook)
        best_idx = sims.argmax()
        return codebook[best_idx], best_idx.item()

    def _init_estimates(self, restart_idx: int) -> list[torch.Tensor]:
        """Initialize factor estimates.

        restart 0: use the bundle (sum) of all codebook items (standard).
        restart 1+: random codebook items.
        """
        if restart_idx == 0:
            # Standard: superposition of all items in each codebook
            return [normalize(cb.sum(dim=0)) for cb in self.codebooks]
        else:
            # Random: pick a random item from each codebook
            return [
                cb[torch.randint(len(cb), (1,)).item()].clone()
                for cb in self.codebooks
            ]

    def factorize_single(
        self, s: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[int], float]:
        """Factorize a single binding (no superposition) with restarts.

        Args:
            s: FHRR vector to factorize, shape (d,)

        Returns:
            (estimated_factors, codebook_indices, best_confidence)
        """
        best_estimates = None
        best_indices = None
        best_conf = -1.0

        for restart in range(self.n_restarts):
            estimates = self._init_estimates(restart)

            for iteration in range(self.max_iters):
                old_estimates = [e.clone() for e in estimates]

                for i in range(self.n_factors):
                    # Unbind s with all OTHER current estimates
                    others = [estimates[j] for j in range(self.n_factors) if j != i]
                    unbound = unbind(s, *others)

                    # Project onto codebook i
                    estimates[i], _ = self._project(unbound, self.codebooks[i])

                # Check convergence
                diffs = [
                    (estimates[i] - old_estimates[i]).abs().max().item()
                    for i in range(self.n_factors)
                ]
                if max(diffs) < self.convergence_threshold:
                    break

            # Score this restart
            reconstructed = bind(*estimates)
            conf = similarity(
                s.unsqueeze(0), reconstructed.unsqueeze(0)
            ).item()

            if conf > best_conf:
                best_conf = conf
                best_estimates = estimates
                best_indices = []
                for i in range(self.n_factors):
                    _, idx = self._project(estimates[i], self.codebooks[i])
                    best_indices.append(idx)

        return best_estimates, best_indices, best_conf

    def factorize_scene(
        self,
        s: torch.Tensor,
        max_objects: int = 5,
        none_shape_idx: int = 3,
        none_color_idx: int = 3,
    ) -> list[tuple[list[int], float]]:
        """Factorize a superposition of bindings via sequential peeling.

        Stops when the resonator recovers a STOP object (shape or color
        index equals the "none" sentinel), eliminating heuristic thresholds.

        Args:
            s: scene FHRR vector (superposition of object bindings)
            max_objects: maximum objects to extract
            none_shape_idx: codebook index for "none" shape (STOP sentinel)
            none_color_idx: codebook index for "none" color (STOP sentinel)

        Returns:
            List of (codebook_indices, confidence) per extracted object.
        """
        residual = s.clone()
        results = []

        for obj_idx in range(max_objects):
            estimates, indices, conf = self.factorize_single(residual)

            # STOP detection: if shape or color is "none", we've hit the
            # sentinel object — stop peeling without including it
            if indices[0] == none_shape_idx or indices[1] == none_color_idx:
                break

            results.append((indices, conf))

            # Peel: subtract the found object's contribution
            reconstructed = bind(*estimates)
            residual = residual - reconstructed

        return results

    def decode_indices(
        self, indices: list[int], codebook_names: list[str] = None
    ) -> dict:
        """Convert codebook indices to a readable dict."""
        if codebook_names is None:
            codebook_names = [f"factor_{i}" for i in range(len(indices))]
        return {name: idx for name, idx in zip(codebook_names, indices)}
