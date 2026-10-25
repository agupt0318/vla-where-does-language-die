"""Reanalyze cached probe activations without any model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.eval.stats import bootstrap_mean
from src.interp.probe import predict_probe


def split_episodes(groups, novel, seed=0):
    ids = np.unique(groups)
    if len(ids) < 4:
        raise ValueError("At least four independent episodes are required")
    ids = np.random.default_rng(seed).permutation(ids)
    train = np.isin(groups, ids[: int(0.7 * len(ids))])
    return (
        np.flatnonzero(train & ~novel),
        np.flatnonzero(~train & ~novel),
        np.flatnonzero(~train & novel),
    )


def analyze(path, seed=0, resamples=10000, shuffles=20):
    if shuffles < 1 or resamples < 1:
        raise ValueError("Positive shuffles and resamples required")
    d = np.load(path, allow_pickle=False)
    y = d["y"]
    selected = d["executed_y"]
    groups = d["groups"]
    novel = d["novel_mask"].astype(bool)
    if not (y.ndim == selected.ndim == groups.ndim == novel.ndim == 1):
        raise ValueError("Labels and groups must be one-dimensional")
    if not (len(y) == len(selected) == len(groups) == len(novel)):
        raise ValueError("Label lengths differ")
    if not np.isin(y, range(4)).all() or not np.isin(selected, range(-1, 4)).all():
        raise ValueError("Invalid destination class")
    tr, te, nv = split_episodes(groups, novel, seed)
    if min(len(tr), len(te), len(nv)) == 0 or set(np.unique(y[tr])) != set(range(4)):
        raise ValueError("Empty split or missing one of four training classes")
    if not np.any(selected[tr] >= 0):
        raise ValueError("No valid training directions for majority baseline")
    if not np.any(selected[te] >= 0):
        raise ValueError("No valid held-out trained directions for behavioral sanity check")
    valid = nv[selected[nv] >= 0]
    if len(np.unique(groups[valid])) < 2:
        raise ValueError("At least two valid evaluation episodes required")

    def estimate(values, indices=valid):
        if len(indices) == 0:
            return None
        return bootstrap_mean(
            values, cluster=groups[indices], seed=seed, resamples=resamples
        ).as_dict()

    majority = int(np.bincount(selected[tr][selected[tr] >= 0], minlength=4).argmax())
    baseline_selected = (selected[valid] == majority).astype(float)
    baseline_named = (y[valid] == majority).astype(float)
    out = {
        "split_seed": seed,
        "bootstrap_resamples": resamples,
        "shuffle_repeats": shuffles,
        "probe_solver_tolerance": 1e-8,
        "n_train": len(tr),
        "n_test_trained": len(te),
        "n_test_novel": len(nv),
        "n_valid_novel": len(valid),
        "n_test_episodes": len(np.unique(groups[valid])),
        "n_invalid_novel": len(nv) - len(valid),
        "n_train_episodes": len(np.unique(groups[tr])),
        "all_training_named_counts": np.bincount(y[tr], minlength=4).tolist(),
        "heldout_trained_named_counts": np.bincount(y[te], minlength=4).tolist(),
        "train_episodes": np.unique(groups[tr]).tolist(),
        "test_episodes": np.unique(groups[nv]).tolist(),
        "novel_named_counts": np.bincount(y[valid], minlength=4).tolist(),
        "novel_selected_counts": np.bincount(selected[valid], minlength=4).tolist(),
        "behavior_trained_accuracy": float(
            np.mean(selected[te[selected[te] >= 0]] == y[te[selected[te] >= 0]])
        ),
        "n_invalid_test_trained": int(np.sum(selected[te] < 0)),
        "behavior_novel_accuracy": float(np.mean(selected[valid] == y[valid])),
        "majority_selected_class_from_training": majority,
        "majority_selected_accuracy": float(baseline_selected.mean()),
        "majority_paired_difference": estimate(baseline_selected - baseline_named),
        "sites": {},
    }
    preds = {
        "train_indices": tr,
        "test_trained_indices": te,
        "test_novel_indices": valid,
        "named": y[valid],
        "selected": selected[valid],
        "groups": groups[valid],
    }
    # The same permuted training labels are used at every site for paired controls.
    shuffled = [np.random.default_rng(seed + 1000 + i).permutation(y[tr]) for i in range(shuffles)]
    for site in sorted(k for k in d.files if k.startswith(("vlm.", "expert."))):
        x = d[site]
        if x.ndim != 2 or x.shape[0] != len(y):
            raise ValueError(f"Activation shape differs from labels: {site}")
        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite activations: {site}")
        pred = predict_probe(x[tr], y[tr], x[valid], seed=seed)
        trained_pred = predict_probe(x[tr], y[tr], x[te], seed=seed)
        named = (pred == y[valid]).astype(float)
        chosen = (pred == selected[valid]).astype(float)
        mismatch = selected[valid] != y[valid]
        null = np.array([predict_probe(x[tr], labels, x[valid], seed=seed) for labels in shuffled])
        null_trained = np.array(
            [predict_probe(x[tr], labels, x[te], seed=seed) for labels in shuffled]
        )
        null_named = (null == y[valid]).mean(axis=1)
        null_chosen = (null == selected[valid]).mean(axis=1)
        out["sites"][site] = {
            "trained_accuracy": float((trained_pred == y[te]).mean()),
            "shuffled_trained_mean": float((null_trained == y[te]).mean()),
            "named": estimate(named),
            "selected": estimate(chosen),
            "delta": estimate(chosen - named),
            "selected_over_majority": estimate(chosen - baseline_selected),
            "selected_over_shuffled": estimate(chosen - (null == selected[valid]).mean(axis=0)),
            "named_over_shuffled": estimate(named - (null == y[valid]).mean(axis=0)),
            "delta_over_shuffled": estimate(
                (chosen - named)
                - ((null == selected[valid]).astype(float) - (null == y[valid]).astype(float)).mean(
                    axis=0
                )
            ),
            "n_disagreement": int(mismatch.sum()),
            "n_disagreement_episodes": int(len(np.unique(groups[valid[mismatch]]))),
            "disagreement_named": estimate(named[mismatch], valid[mismatch]),
            "disagreement_selected": estimate(chosen[mismatch], valid[mismatch]),
            "disagreement_delta": estimate((chosen - named)[mismatch], valid[mismatch]),
            "shuffled_named_mean": float(null_named.mean()),
            "shuffled_selected_mean": float(null_chosen.mean()),
            "shuffled_delta_mean": float((null_chosen - null_named).mean()),
            "shuffled_delta_range": np.quantile(null_chosen - null_named, [0.025, 0.975]).tolist(),
        }
        preds[site] = pred
        preds[site + "__null"] = null
        preds[site + "__trained"] = trained_pred
        preds[site + "__null_trained"] = null_trained
    np.savez_compressed(path.parent / f"predictions_seed{seed}.npz", **preds)
    target = path.parent / f"analysis_seed{seed}.json"
    target.write_text(json.dumps(out, indent=2, allow_nan=False))
    print(target)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cache", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resamples", type=int, default=10000)
    p.add_argument("--shuffles", type=int, default=20)
    a = p.parse_args()
    analyze(a.cache, a.seed, a.resamples, a.shuffles)
