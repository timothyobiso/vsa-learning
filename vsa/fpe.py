"""Fractional Power Encoding (FPE) for continuous/ordinal values.

Given a base vector b = exp(iθ), the FPE for value p ∈ [0, 1] is:
    FPE(b, p) = exp(i * p * θ) = b^p

This creates a smooth encoding where nearby values have similar vectors.
"""

from typing import Union

import torch


def fpe_encode(base: torch.Tensor, power: Union[float, torch.Tensor]) -> torch.Tensor:
    """Encode a continuous value via fractional power of a base vector.

    Args:
        base: FHRR base vector, shape (d,) or (B, d)
        power: scalar or tensor of fractional powers in [0, 1]

    Returns:
        FPE-encoded vector(s) on the unit circle.
    """
    phases = torch.angle(base)
    if isinstance(power, torch.Tensor) and power.dim() > 0:
        # power shape: (N,) or (B,) → broadcast with phases
        power = power.unsqueeze(-1)  # (N, 1)
        if phases.dim() == 1:
            phases = phases.unsqueeze(0)  # (1, d)
    return torch.exp(1j * power * phases)


def fpe_codebook(base: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Create a codebook of n_levels FPE vectors from a base vector.

    Levels are uniformly spaced in [0, 1].

    Args:
        base: FHRR base vector, shape (d,)
        n_levels: number of discretization levels

    Returns:
        Codebook tensor of shape (n_levels, d) on the unit circle.
    """
    powers = torch.linspace(0, 1, n_levels)
    return fpe_encode(base, powers)
