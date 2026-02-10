"""Scene codebook management.

Creates and stores fixed codebooks for scene properties.
Config-driven: supports arbitrary factor configurations for different datasets.

Each factor is either:
- discrete: random FHRR vectors for each category + "none" sentinel (last index)
- fpe: Fractional Power Encoding for continuous values in [0,1]
"""

import torch
from dataclasses import dataclass, field
from .fhrr import random_vectors
from .fpe import fpe_codebook


@dataclass
class FactorConfig:
    """Configuration for a single factor in a scene codebook.

    Args:
        name: factor name (e.g. "shape", "color", "x")
        type: "discrete" or "fpe"
        values: list of category names (discrete only). A "none" sentinel is appended automatically.
        n_levels: number of FPE discretization levels (fpe only)
    """
    name: str
    type: str  # "discrete" or "fpe"
    values: list[str] = field(default_factory=list)
    n_levels: int = 32


# ── Dataset-specific constants ──────────────────────────────────────────────

TOY_SHAPES = ["circle", "square", "triangle"]
TOY_COLORS = ["red", "green", "blue"]

CLEVR_SHAPES = ["cube", "sphere", "cylinder"]
CLEVR_COLORS = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
CLEVR_MATERIALS = ["metal", "rubber"]

DSPRITES_SHAPES = ["ellipse", "heart", "square"]

# Backward-compatible aliases
SHAPES = TOY_SHAPES + ["none"]
COLORS = TOY_COLORS + ["none"]


class SceneCodebooks:
    """Fixed codebooks for scene factorization, driven by a list of FactorConfigs."""

    def __init__(self, d: int, factors: list[FactorConfig], seed: int = 42):
        self.d = d
        self.factors = factors

        gen = torch.Generator().manual_seed(seed)

        # Storage for codebooks, value-to-index mappings, and FPE bases
        self._codebooks: list[torch.Tensor] = []
        self._value_to_idx: dict[str, dict[str, int]] = {}
        self._fpe_bases: dict[str, torch.Tensor] = {}
        self._fpe_n_levels: dict[str, int] = {}

        for fc in factors:
            if fc.type == "discrete":
                # +1 for "none" sentinel (appended as last index)
                n_vecs = len(fc.values) + 1
                vecs = random_vectors(n_vecs, d, generator=gen)
                self._codebooks.append(vecs)
                # Map value names → indices; "none" is last
                mapping = {v: i for i, v in enumerate(fc.values)}
                mapping["none"] = len(fc.values)
                self._value_to_idx[fc.name] = mapping
            elif fc.type == "fpe":
                base = random_vectors(1, d, generator=gen).squeeze(0)
                cb = fpe_codebook(base, fc.n_levels)
                self._codebooks.append(cb)
                self._fpe_bases[fc.name] = base
                self._fpe_n_levels[fc.name] = fc.n_levels
            else:
                raise ValueError(f"Unknown factor type: {fc.type}")

    # ── Factory classmethods ────────────────────────────────────────────

    @classmethod
    def toy(cls, d: int, n_pos_levels: int = 32, n_size_levels: int = 16, seed: int = 42) -> "SceneCodebooks":
        """5-factor toy setup: shape(3), color(3), x, y, size."""
        factors = [
            FactorConfig("shape", "discrete", TOY_SHAPES),
            FactorConfig("color", "discrete", TOY_COLORS),
            FactorConfig("x", "fpe", n_levels=n_pos_levels),
            FactorConfig("y", "fpe", n_levels=n_pos_levels),
            FactorConfig("size", "fpe", n_levels=n_size_levels),
        ]
        return cls(d, factors, seed=seed)

    @classmethod
    def clevr(cls, d: int, n_pos_levels: int = 32, n_size_levels: int = 16, seed: int = 42) -> "SceneCodebooks":
        """7-factor CLEVR setup: shape(3), color(8), material(2), x, y, z, size."""
        factors = [
            FactorConfig("shape", "discrete", CLEVR_SHAPES),
            FactorConfig("color", "discrete", CLEVR_COLORS),
            FactorConfig("material", "discrete", CLEVR_MATERIALS),
            FactorConfig("x", "fpe", n_levels=n_pos_levels),
            FactorConfig("y", "fpe", n_levels=n_pos_levels),
            FactorConfig("z", "fpe", n_levels=n_pos_levels),
            FactorConfig("size", "fpe", n_levels=n_size_levels),
        ]
        return cls(d, factors, seed=seed)

    @classmethod
    def dsprites(cls, d: int, n_pos_levels: int = 32, n_size_levels: int = 16, seed: int = 42) -> "SceneCodebooks":
        """7-factor dSprites setup: shape(3), R, G, B, x, y, scale."""
        factors = [
            FactorConfig("shape", "discrete", DSPRITES_SHAPES),
            FactorConfig("R", "fpe", n_levels=n_size_levels),
            FactorConfig("G", "fpe", n_levels=n_size_levels),
            FactorConfig("B", "fpe", n_levels=n_size_levels),
            FactorConfig("x", "fpe", n_levels=n_pos_levels),
            FactorConfig("y", "fpe", n_levels=n_pos_levels),
            FactorConfig("scale", "fpe", n_levels=n_size_levels),
        ]
        return cls(d, factors, seed=seed)

    # ── Backward-compatible properties ──────────────────────────────────

    @property
    def n_pos_levels(self) -> int:
        return self._fpe_n_levels.get("x", 32)

    @property
    def n_size_levels(self) -> int:
        return self._fpe_n_levels.get("size", self._fpe_n_levels.get("scale", 16))

    @property
    def shape_to_idx(self) -> dict[str, int]:
        return self._value_to_idx["shape"]

    @property
    def color_to_idx(self) -> dict[str, int]:
        return self._value_to_idx.get("color", {})

    # ── Core API ────────────────────────────────────────────────────────

    def to(self, device: torch.device) -> "SceneCodebooks":
        """Move all codebooks to a device."""
        self._codebooks = [cb.to(device) for cb in self._codebooks]
        self._fpe_bases = {k: v.to(device) for k, v in self._fpe_bases.items()}
        return self

    def all_codebooks(self) -> list[torch.Tensor]:
        """Return list of all codebooks for the resonator."""
        return list(self._codebooks)

    def codebook_names(self) -> list[str]:
        return [fc.name for fc in self.factors]

    def discrete_sentinel_indices(self) -> list[tuple[int, int]]:
        """Return (factor_index, sentinel_codebook_index) for each discrete factor.

        Used by the resonator for STOP detection.
        """
        result = []
        for i, fc in enumerate(self.factors):
            if fc.type == "discrete":
                sentinel_idx = len(fc.values)  # "none" is the last entry
                result.append((i, sentinel_idx))
        return result

    @property
    def stop_vector(self) -> torch.Tensor:
        """STOP object: bind of all sentinels (none for discrete, index 0 for FPE)."""
        from .fhrr import bind

        sentinel_parts = []
        for i, fc in enumerate(self.factors):
            if fc.type == "discrete":
                sentinel_parts.append(self._codebooks[i][len(fc.values)])  # "none"
            else:
                sentinel_parts.append(self._codebooks[i][0])  # FPE index 0
        return bind(*sentinel_parts)

    def encode_object(self, **props) -> torch.Tensor:
        """Encode a single object as a bound FHRR vector.

        Args:
            **props: property dict with keys matching factor names.
                     Discrete factors: pass the category name (str).
                     FPE factors: pass a float in [0, 1].
        """
        from .fhrr import bind
        from .fpe import fpe_encode

        parts = []
        for i, fc in enumerate(self.factors):
            val = props[fc.name]
            if fc.type == "discrete":
                idx = self._value_to_idx[fc.name][val]
                parts.append(self._codebooks[i][idx])
            else:
                base = self._fpe_bases[fc.name]
                parts.append(fpe_encode(base, val))
        return bind(*parts)

    def encode_scene(self, objects: list[dict]) -> torch.Tensor:
        """Encode a full scene as a bundled superposition of object bindings.

        Args:
            objects: list of dicts with keys matching factor names
        """
        from .fhrr import bundle

        bindings = [self.encode_object(**obj) for obj in objects]
        scene = bundle(*bindings)
        scene = scene + 0.5 * self.stop_vector
        return scene

    def nearest_fpe_idx(self, factor_name: str, value: float) -> int:
        """Map a [0,1] value to the nearest FPE codebook index for a given factor."""
        n_levels = self._fpe_n_levels[factor_name]
        return round(value * (n_levels - 1))

    def factor_type(self, factor_name: str) -> str:
        """Return "discrete" or "fpe" for a given factor."""
        for fc in self.factors:
            if fc.name == factor_name:
                return fc.type
        raise KeyError(f"Unknown factor: {factor_name}")

    def factor_values(self, factor_name: str) -> list[str]:
        """Return the list of discrete values for a factor (excluding 'none')."""
        for fc in self.factors:
            if fc.name == factor_name:
                if fc.type != "discrete":
                    raise ValueError(f"Factor '{factor_name}' is FPE, not discrete")
                return list(fc.values)
        raise KeyError(f"Unknown factor: {factor_name}")

    def object_to_indices(self, obj: dict) -> list[int]:
        """Convert an object property dict to a list of codebook indices.

        Args:
            obj: dict with keys matching factor names.
                 Discrete factors: str category name.
                 FPE factors: float in [0, 1].

        Returns:
            List of K ints, one codebook index per factor.
        """
        indices = []
        for fc in self.factors:
            if fc.type == "discrete":
                indices.append(self._value_to_idx[fc.name][obj[fc.name]])
            else:
                indices.append(self.nearest_fpe_idx(fc.name, obj[fc.name]))
        return indices

    def stop_indices(self) -> list[int]:
        """Return the codebook indices for the STOP object.

        Discrete factors use the sentinel ("none") index; FPE factors use index 0.
        """
        indices = []
        for fc in self.factors:
            if fc.type == "discrete":
                indices.append(len(fc.values))  # "none" sentinel
            else:
                indices.append(0)  # FPE index 0
        return indices
