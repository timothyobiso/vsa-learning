"""Sanity check: verify the VSA pipeline works without a neural network.

1. Encode a scene algebraically (ground truth FHRR vector)
2. Run the resonator to factorize it
3. Check if we recover the correct object properties

If this doesn't work, the neural encoder has no chance.
"""

import torch
from vsa.codebooks import SceneCodebooks, SHAPES, COLORS
from vsa.resonator import ResonatorNetwork
from vsa.fhrr import similarity


def main():
    d = 2048
    codebooks = SceneCodebooks(d=d, n_pos_levels=32, n_size_levels=16)
    resonator = ResonatorNetwork(codebooks.all_codebooks(), max_iters=200)

    print(f"=== Sanity Check: FHRR + FPE + Resonator ===")
    print(f"Dimension: {d}")
    print(f"Codebooks: {[cb.shape[0] for cb in codebooks.all_codebooks()]}")
    print()

    # --- Test 1: Single object ---
    print("--- Test 1: Single object ---")
    obj = {"shape": "triangle", "color": "blue", "x": 0.3, "y": 0.7, "size": 0.5}
    scene_vec = codebooks.encode_scene([obj])

    results = resonator.factorize_scene(scene_vec, max_objects=3)
    print(f"  Ground truth: {obj}")
    print(f"  Recovered {len(results)} object(s) (STOP sentinel terminated peeling)")
    for indices, conf in results:
        decoded = {
            "shape": SHAPES[indices[0]],
            "color": COLORS[indices[1]],
            "x_idx": indices[2],
            "y_idx": indices[3],
            "size_idx": indices[4],
        }
        gt_x_idx = round(obj["x"] * 31)
        gt_y_idx = round(obj["y"] * 31)
        gt_size_idx = round(obj["size"] * 15)
        print(f"  Recovered:    {decoded} (conf={conf:.3f})")
        print(f"  GT indices:   x={gt_x_idx}, y={gt_y_idx}, size={gt_size_idx}")

        shape_ok = decoded["shape"] == obj["shape"]
        color_ok = decoded["color"] == obj["color"]
        x_ok = abs(indices[2] - gt_x_idx) <= 1
        y_ok = abs(indices[3] - gt_y_idx) <= 1
        size_ok = abs(indices[4] - gt_size_idx) <= 1
        print(f"  Correct: shape={shape_ok} color={color_ok} x={x_ok} y={y_ok} size={size_ok}")
    print()

    # --- Test 2: Two objects ---
    print("--- Test 2: Two objects (superposition) ---")
    objects = [
        {"shape": "circle", "color": "red", "x": 0.2, "y": 0.2, "size": 0.3},
        {"shape": "square", "color": "green", "x": 0.8, "y": 0.6, "size": 0.7},
    ]
    scene_vec = codebooks.encode_scene(objects)

    results = resonator.factorize_scene(scene_vec, max_objects=5)
    print(f"  Ground truth:")
    for o in objects:
        print(f"    {o}")
    print(f"  Recovered {len(results)} object(s) (STOP sentinel terminated peeling):")
    for indices, conf in results:
        decoded = {
            "shape": SHAPES[indices[0]],
            "color": COLORS[indices[1]],
            "x_idx": indices[2],
            "y_idx": indices[3],
            "size_idx": indices[4],
        }
        print(f"    {decoded} (conf={conf:.3f})")
    print()

    # --- Test 3: Three objects (stress test) ---
    print("--- Test 3: Three objects (stress test) ---")
    objects3 = [
        {"shape": "triangle", "color": "blue", "x": 0.1, "y": 0.9, "size": 0.2},
        {"shape": "circle", "color": "green", "x": 0.5, "y": 0.5, "size": 0.5},
        {"shape": "square", "color": "red", "x": 0.9, "y": 0.1, "size": 0.8},
    ]
    scene_vec = codebooks.encode_scene(objects3)

    results = resonator.factorize_scene(scene_vec, max_objects=7)
    print(f"  Ground truth: {len(objects3)} objects")
    for o in objects3:
        print(f"    {o}")
    print(f"  Recovered {len(results)} object(s) (STOP sentinel terminated peeling):")
    for indices, conf in results:
        decoded = {
            "shape": SHAPES[indices[0]],
            "color": COLORS[indices[1]],
            "x_idx": indices[2],
            "y_idx": indices[3],
            "size_idx": indices[4],
        }
        print(f"    {decoded} (conf={conf:.3f})")


if __name__ == "__main__":
    main()
