"""Verify the exact experimental checkpoint without importing XGBoost or pytest.

Checks upstream parity, non-perturbing activation capture, determinism, and an
independent recomputation of the motion label on the first stored observation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.reproduce_ifr import STATE_KEYS, build_images, build_state  # noqa: E402
from src.eval.composition import DESTINATIONS, instruction_for  # noqa: E402
from src.interp.probe import pool_activation  # noqa: E402
from src.models.smolvla import SmolVLA, make_batch  # noqa: E402


def verify(run: Path) -> dict:
    config = yaml.safe_load((run / "config.yaml").read_text())
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    metadata = json.loads((run / "states.json").read_text())[0]
    cache_path = run / "activations.npz"
    if not cache_path.exists():
        cache_path = run / "partial_activations.npz"
    cache = np.load(cache_path, allow_pickle=False)
    model = SmolVLA.load(config["checkpoint"], device="cpu")
    assert model.has_norm_stats, "Checkpoint normalization statistics missing"
    with h5py.File(ROOT / "data/libero/libero_goal" / f"{metadata['task']}_demo.hdf5") as h:
        obs = {
            key: h["data"][metadata["demo"]]["obs"][key][metadata["t"]]
            for key in (*STATE_KEYS, "agentview_rgb", "eye_in_hand_rgb")
        }
    keys = list(model.config.image_features)
    images = build_images(obs, keys, model.config.image_features[keys[0]].shape[-1])
    state = model.normalize_state(build_state(obs, model.config.robot_state_feature.shape[0]))
    batch = make_batch(
        images, state, instruction_for("bowl", next(iter(DESTINATIONS))), model.policy, "cpu"
    )
    sites = json.loads((run / "environment.json").read_text())["sites"]
    noise = model.make_noise(1)
    model.policy.reset()
    with torch.inference_mode():
        stock = model.policy.predict_action_chunk(dict(batch), noise=noise)
        captured = model.forward_with_cache(batch, sites=sites, noise=noise)
        repeat = model.forward_with_cache(batch, sites=sites, noise=noise)
    assert torch.equal(stock, captured.action), "Instrumentation changed upstream output"
    assert torch.equal(captured.action, repeat.action), "Fixed-noise inference differs"
    errors = {}
    for site in sites:
        occurrences = captured.occurrences(site)
        expected = 1 if site.startswith("vlm.") else model.config.num_steps
        assert len(occurrences) == expected, (site, len(occurrences), expected)
        pooled = pool_activation(occurrences[0][0].float().cpu().numpy())
        errors[site] = float(np.max(np.abs(pooled - cache[site][0])))
        np.testing.assert_allclose(pooled, cache[site][0], rtol=1e-6, atol=1e-7)
    action = model.unnormalize_action(captured.action)[0].cpu().numpy()
    displacement = action[:, :3].sum(axis=0).astype(float)
    ee = np.asarray(obs["ee_pos"], dtype=float)
    # Independent implementation, deliberately does not call score_direction.
    scores = []
    for dest in DESTINATIONS:
        direction = np.asarray(config["anchors"][dest], dtype=float) - ee
        scores.append(
            float(
                np.dot(direction, displacement)
                / (np.linalg.norm(direction) * np.linalg.norm(displacement))
            )
        )
    label = int(np.argmax(scores)) if np.linalg.norm(displacement) >= 1e-6 else -1
    assert label == cache["executed_y"][0], "Motion label differs from saved inference"
    result = {
        "checkpoint": config["checkpoint"],
        "state": metadata,
        "upstream_parity": True,
        "capture_is_non_perturbing": True,
        "fixed_noise_determinism": True,
        "occurrence_counts_correct": True,
        "replayed_pooled_activation_max_errors": errors,
        "independent_motion_label_matches_cache": True,
        "action": action.tolist(),
        "summed_translation": displacement.tolist(),
        "anchor_cosines": scores,
        "selected_label": label,
    }
    (run / "inference_verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    verify(parser.parse_args().run)
