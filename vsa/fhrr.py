"""FHRR (Fourier Holographic Reduced Representation) operations.

Vectors are complex-valued with unit magnitude: z = exp(iθ).
Binding is element-wise complex multiplication.
Bundling is element-wise addition (then normalize).
"""

import torch
import torch.nn.functional as F


def random_vectors(n: int, d: int, generator=None) -> torch.Tensor:
    """Generate n random FHRR vectors of dimension d on the complex unit circle."""
    phases = torch.rand(n, d, generator=generator) * 2 * torch.pi
    return torch.exp(1j * phases)


def normalize(x: torch.Tensor) -> torch.Tensor:
    """Project complex vector onto the unit circle (phase-only)."""
    return x / (x.abs().clamp(min=1e-8))


def bind(*vectors: torch.Tensor) -> torch.Tensor:
    """Bind vectors via element-wise complex multiplication."""
    result = vectors[0]
    for v in vectors[1:]:
        result = result * v
    return result


def unbind(bound: torch.Tensor, *keys: torch.Tensor) -> torch.Tensor:
    """Unbind by multiplying with conjugates of key vectors."""
    result = bound
    for k in keys:
        result = result * k.conj()
    return result


def bundle(*vectors: torch.Tensor) -> torch.Tensor:
    """Bundle vectors via element-wise addition."""
    result = vectors[0]
    for v in vectors[1:]:
        result = result + v
    return result


def similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine similarity between complex vectors.

    Returns real-valued similarity in [-1, 1].
    Supports batched inputs: a (B, d) vs b (N, d) → (B, N).
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)

    a_norm = a / a.abs().clamp(min=1e-8)
    b_norm = b / b.abs().clamp(min=1e-8)

    # (B, d) x (d, N) → (B, N)
    dots = torch.mm(a_norm, b_norm.conj().T)
    return dots.real / a.shape[-1]
