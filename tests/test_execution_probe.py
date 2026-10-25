import numpy as np
import pytest

from scripts.analyze_execution_probe import analyze, split_episodes


def test_split_keeps_sibling_observations_out_of_training():
    groups = np.repeat(np.arange(20), 8)
    novel = np.tile([False] * 4 + [True] * 4, 20)
    tr, te, nv = split_episodes(groups, novel)
    assert set(groups[tr]).isdisjoint(groups[te])
    assert set(groups[tr]).isdisjoint(groups[nv])
    assert not novel[tr].any()
    assert novel[nv].all()


def test_too_few_episodes_fails_closed():
    with pytest.raises(ValueError, match="four independent"):
        split_episodes(np.array([0, 0, 1, 1]), np.array([False, True] * 2))


def test_known_instruction_encoding_has_negative_motion_alignment(tmp_path):
    groups = np.repeat(np.arange(20), 8)
    y = np.tile(np.arange(4), 40)
    novel = np.tile([False] * 4 + [True] * 4, 20)
    selected = np.where(novel, (y + 1) % 4, y)
    path = tmp_path / "activations.npz"
    np.savez(
        path,
        y=y,
        executed_y=selected,
        groups=groups,
        novel_mask=novel,
        **{"vlm.L0.resid_post": np.eye(4)[y]},
    )
    result = analyze(path, resamples=100, shuffles=2)
    row = result["sites"]["vlm.L0.resid_post"]
    assert row["named"]["value"] == 1
    assert row["selected"]["value"] == 0
    assert row["delta"]["value"] == -1
    assert row["delta"]["lo"] == -1 and row["delta"]["hi"] == -1
    assert row["n_disagreement"] == result["n_valid_novel"]


def test_invalid_action_labels_are_rejected(tmp_path):
    path = tmp_path / "activations.npz"
    np.savez(
        path,
        y=np.arange(4),
        executed_y=np.array([0, 1, np.nan, 3]),
        groups=np.arange(4),
        novel_mask=np.array([0, 0, 1, 1]),
    )
    with pytest.raises(ValueError, match="Invalid destination"):
        analyze(path, resamples=10, shuffles=2)


def test_invalid_motion_is_excluded_from_both_accuracy_arms(tmp_path):
    groups = np.repeat(np.arange(20), 8)
    y = np.tile(np.arange(4), 40)
    novel = np.tile([False] * 4 + [True] * 4, 20)
    selected = y.copy()
    selected[novel & (y == 0)] = -1
    path = tmp_path / "activations.npz"
    np.savez(
        path,
        y=y,
        executed_y=selected,
        groups=groups,
        novel_mask=novel,
        **{"vlm.L0.resid_post": np.eye(4)[y]},
    )
    result = analyze(path, resamples=100, shuffles=2)
    row = result["sites"]["vlm.L0.resid_post"]
    assert result["n_invalid_novel"] > 0
    assert row["named"]["n"] == row["selected"]["n"] == result["n_valid_novel"]
    assert row["named"]["value"] == row["selected"]["value"] == 1
    assert row["delta"]["value"] == 0
    assert row["disagreement_delta"] is None


def test_constant_decoder_motion_preference_is_explained_by_controls(tmp_path):
    groups = np.repeat(np.arange(20), 8)
    y = np.tile(np.arange(4), 40)
    novel = np.tile([False] * 4 + [True] * 4, 20)
    path = tmp_path / "activations.npz"
    np.savez(
        path,
        y=y,
        executed_y=np.zeros(len(y), dtype=int),
        groups=groups,
        novel_mask=novel,
        **{"vlm.L0.resid_post": np.zeros((len(y), 3))},
    )
    result = analyze(path, resamples=100, shuffles=2)
    row = result["sites"]["vlm.L0.resid_post"]
    assert row["delta"]["value"] > 0
    assert row["selected_over_majority"]["value"] == 0
    assert row["selected_over_shuffled"]["value"] == 0
