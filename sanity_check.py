"""Sanity check: verify the VSA pipeline works without a neural network.

1. Encode a scene algebraically (ground truth FHRR vector)
2. Run the resonator to factorize it
3. Check if we recover the correct object properties

If this doesn't work, the neural encoder has no chance.
"""

import argparse
import torch
from vsa.codebooks import (
    SceneCodebooks, SHAPES, COLORS,
    CLEVR_SHAPES, CLEVR_COLORS, CLEVR_MATERIALS,
    DSPRITES_SHAPES,
)
from vsa.resonator import ResonatorNetwork
from vsa.fhrr import similarity


def decode_indices(codebooks: SceneCodebooks, indices: list[int]) -> dict:
    """Convert codebook indices to a readable dict using factor configs."""
    result = {}
    for i, fc in enumerate(codebooks.factors):
        if fc.type == "discrete":
            # Reverse lookup: index → value name
            idx_to_val = {v: k for k, v in codebooks._value_to_idx[fc.name].items()}
            result[fc.name] = idx_to_val.get(indices[i], f"idx={indices[i]}")
        else:
            result[f"{fc.name}_idx"] = indices[i]
    return result


def check_object(
    codebooks: SceneCodebooks, obj: dict, indices: list[int], tolerance: int = 1
) -> dict[str, bool]:
    """Check if recovered indices match ground-truth object properties."""
    results = {}
    for i, fc in enumerate(codebooks.factors):
        if fc.type == "discrete":
            gt_idx = codebooks._value_to_idx[fc.name][obj[fc.name]]
            results[fc.name] = indices[i] == gt_idx
        else:
            gt_idx = codebooks.nearest_fpe_idx(fc.name, obj[fc.name])
            results[fc.name] = abs(indices[i] - gt_idx) <= tolerance
    return results


def run_test(
    codebooks: SceneCodebooks,
    resonator: ResonatorNetwork,
    objects: list[dict],
    test_name: str,
    max_peel: int = 5,
):
    """Run a single encode → factorize test."""
    sentinel_pairs = codebooks.discrete_sentinel_indices()
    scene_vec = codebooks.encode_scene(objects)

    results = resonator.factorize_scene(
        scene_vec, max_objects=max_peel, sentinel_pairs=sentinel_pairs,
    )

    print(f"--- {test_name} ---")
    print(f"  Ground truth: {len(objects)} object(s)")
    for o in objects:
        print(f"    {o}")
    print(f"  Recovered {len(results)} object(s) (STOP sentinel terminated peeling):")
    for indices, conf in results:
        decoded = decode_indices(codebooks, indices)
        print(f"    {decoded} (conf={conf:.3f})")
    print()


def test_toy(d: int = 2048):
    """Run toy dataset sanity checks."""
    codebooks = SceneCodebooks.toy(d=d, n_pos_levels=32, n_size_levels=16)
    resonator = ResonatorNetwork(codebooks.all_codebooks(), max_iters=200)

    print(f"=== Toy Dataset Sanity Check ===")
    print(f"Dimension: {d}")
    print(f"Factors: {codebooks.codebook_names()}")
    print(f"Codebook sizes: {[cb.shape[0] for cb in codebooks.all_codebooks()]}")
    print()

    # Test 1: Single object
    run_test(codebooks, resonator, [
        {"shape": "triangle", "color": "blue", "x": 0.3, "y": 0.7, "size": 0.5},
    ], "Test 1: Single object")

    # Test 2: Two objects
    run_test(codebooks, resonator, [
        {"shape": "circle", "color": "red", "x": 0.2, "y": 0.2, "size": 0.3},
        {"shape": "square", "color": "green", "x": 0.8, "y": 0.6, "size": 0.7},
    ], "Test 2: Two objects (superposition)")

    # Test 3: Three objects
    run_test(codebooks, resonator, [
        {"shape": "triangle", "color": "blue", "x": 0.1, "y": 0.9, "size": 0.2},
        {"shape": "circle", "color": "green", "x": 0.5, "y": 0.5, "size": 0.5},
        {"shape": "square", "color": "red", "x": 0.9, "y": 0.1, "size": 0.8},
    ], "Test 3: Three objects (stress test)", max_peel=7)


def test_clevr(d: int = 2048):
    """Run CLEVR codebook sanity checks."""
    codebooks = SceneCodebooks.clevr(d=d, n_pos_levels=32, n_size_levels=16)
    resonator = ResonatorNetwork(codebooks.all_codebooks(), max_iters=200)

    print(f"=== CLEVR Dataset Sanity Check ===")
    print(f"Dimension: {d}")
    print(f"Factors: {codebooks.codebook_names()}")
    print(f"Codebook sizes: {[cb.shape[0] for cb in codebooks.all_codebooks()]}")
    print()

    # Test 1: Single CLEVR object
    run_test(codebooks, resonator, [
        {"shape": "sphere", "color": "red", "material": "metal",
         "x": 0.3, "y": 0.7, "z": 0.5, "size": 0.25},
    ], "Test 1: Single CLEVR object")

    # Test 2: Two CLEVR objects
    run_test(codebooks, resonator, [
        {"shape": "cube", "color": "blue", "material": "rubber",
         "x": 0.2, "y": 0.3, "z": 0.1, "size": 0.75},
        {"shape": "cylinder", "color": "yellow", "material": "metal",
         "x": 0.8, "y": 0.6, "z": 0.4, "size": 0.25},
    ], "Test 2: Two CLEVR objects")


def test_dsprites(d: int = 2048):
    """Run dSprites codebook sanity checks."""
    codebooks = SceneCodebooks.dsprites(d=d, n_pos_levels=32, n_size_levels=16)
    resonator = ResonatorNetwork(codebooks.all_codebooks(), max_iters=200)

    print(f"=== dSprites Dataset Sanity Check ===")
    print(f"Dimension: {d}")
    print(f"Factors: {codebooks.codebook_names()}")
    print(f"Codebook sizes: {[cb.shape[0] for cb in codebooks.all_codebooks()]}")
    print()

    # Test 1: Single dSprites object
    run_test(codebooks, resonator, [
        {"shape": "ellipse", "R": 0.8, "G": 0.2, "B": 0.5,
         "x": 0.4, "y": 0.6, "scale": 0.5},
    ], "Test 1: Single dSprites object")

    # Test 2: Two dSprites objects
    run_test(codebooks, resonator, [
        {"shape": "heart", "R": 1.0, "G": 0.0, "B": 0.0,
         "x": 0.2, "y": 0.3, "scale": 0.3},
        {"shape": "square", "R": 0.0, "G": 0.0, "B": 1.0,
         "x": 0.7, "y": 0.8, "scale": 0.7},
    ], "Test 2: Two dSprites objects")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="toy",
                        choices=["toy", "clevr", "dsprites", "all"],
                        help="Which dataset config to test")
    parser.add_argument("--dim", type=int, default=2048, help="FHRR dimension")
    args = parser.parse_args()

    if args.dataset == "toy" or args.dataset == "all":
        test_toy(args.dim)
    if args.dataset == "clevr" or args.dataset == "all":
        test_clevr(args.dim)
    if args.dataset == "dsprites" or args.dataset == "all":
        test_dsprites(args.dim)


if __name__ == "__main__":
    main()
