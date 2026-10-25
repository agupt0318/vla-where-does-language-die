#!/usr/bin/env python
"""Probe layer activations against both named and action-selected destinations.

    python scripts/run_probe.py --checkpoint k1000dai/smolvla_libero_finetune --device cuda

The existing named-destination probe remains intact. This run also scores the same probe
prediction on novel trials against the destination selected by the policy's predicted
motion. This comparison is correlational; it does not localize a causal mechanism.

Cheap by construction. One forward pass per (state, instruction), activations cached for
every site at once, then probes fit on CPU. No per-site forward passes, so this costs a
small fraction of the M2 sweep.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.reproduce_ifr import STATE_KEYS, build_images, build_state  # noqa: E402
from scripts.run_composition import collect_states, task_endpoints  # noqa: E402
from src.eval.composition import (  # noqa: E402
    DESTINATIONS,
    OBJECT_SOURCE_TASKS,
    TRAINED,
    build_anchors,
    instruction_for,
    score_direction,
)
from src.eval.stats import bootstrap_mean  # noqa: E402
from src.interp.probe import (  # noqa: E402
    ProbeResult,
    fit_probe,
    pool_activation,
    predict_probe,
)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _fit_nonlinear_in_subprocess(
    out_dir: Path,
    site_names: list[str],
    acts: dict[str, list[np.ndarray]],
    y: np.ndarray,
    tr: np.ndarray,
    te: np.ndarray,
    nv: np.ndarray,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Fit XGBoost probes in a fresh process.

    On macOS/arm64, XGBoost can segfault if it receives NumPy labels after Torch/MPS has
    already been active in the process. Keeping the boosted trees in a CPU-only child
    preserves the probe definition while letting the parent own the Metal forward pass.
    """
    input_path = out_dir / "nonlinear_input.npz"
    output_path = out_dir / "nonlinear_metrics.json"
    np.savez_compressed(
        input_path,
        y=y,
        tr=tr.astype(int),
        te=te.astype(int),
        nv=nv.astype(int),
        site_names=np.asarray(site_names),
        **{name: np.stack(acts[name]) for name in site_names},
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.fit_probe_nonlinear",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--seed",
            str(seed),
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="libero_goal")
    ap.add_argument("--data-root", type=Path, default=REPO_ROOT / "data" / "libero")
    ap.add_argument("--n-states", type=int, default=60)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resamples", type=int, default=10_000)
    ap.add_argument("--sites-limit", type=int, default=None)
    ap.add_argument("--run-id", default=None)
    ap.add_argument(
        "--resume-from", type=Path, help="verified prefix run with activations and state metadata"
    )
    ap.add_argument(
        "--nonlinear",
        action="store_true",
        help=(
            "also fit a gradient-boosted-tree probe at every site, on the same split. "
            "Distinguishes 'the destination is not encoded here' from 'it is encoded but "
            "not linearly readable' -- the two readings the linear null cannot separate."
        ),
    )
    ap.add_argument(
        "--cache-activations",
        action="store_true",
        help="write pooled activations to activations.npz so later probes need no forward passes",
    )
    args = ap.parse_args()
    if args.n_states < 4 or args.resamples < 1:
        ap.error("n-states must be >=4 and resamples >=1")

    suite_dir = args.data_root / args.suite
    run_id = args.run_id or f"probe_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = REPO_ROOT / "results" / run_id
    if args.resume_from and args.resume_from.resolve() == out_dir.resolve():
        ap.error("Resume into a new run-id so the source provenance remains unchanged")
    if (out_dir / "activations.npz").exists():
        ap.error("Run-id already has a completed activation cache; choose a new run-id")
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from src.models.smolvla import SmolVLA, make_batch

    print(f"run_id     : {run_id}\ncheckpoint : {args.checkpoint}\ndevice     : {args.device}")
    model = SmolVLA.load(args.checkpoint, device=args.device)
    cfg = model.config
    keys = list(cfg.image_features)
    size = cfg.image_features[keys[0]].shape[-1]
    sdim = cfg.robot_state_feature.shape[0]

    sites = [
        s
        for s in model.sites()
        if s.name
        in {
            f"{tower}.L{layer}.resid_post" for tower in ("vlm", "expert") for layer in (0, 4, 8, 12)
        }
    ]
    if args.sites_limit:
        sites = sites[: args.sites_limit]
    site_names = [s.name for s in sites]
    dests = list(DESTINATIONS)
    chance = 1.0 / len(dests)
    anchors = build_anchors(task_endpoints(suite_dir))
    if len(anchors) != len(dests):
        raise RuntimeError(f"expected {len(dests)} destination anchors, found {len(anchors)}")
    if not all(np.isfinite(a).all() for a in anchors.values()):
        raise RuntimeError("Non-finite destination anchor")
    print(f"sites      : {len(sites)}   destinations: {len(dests)}   chance: {chance:.3f}")

    (out_dir / "config.yaml").write_text(
        "\n".join(
            f"{k}: {json.dumps(v)}"
            for k, v in {
                "run_id": run_id,
                "experiment": "layerwise_execution_destination_probe",
                "checkpoint": args.checkpoint,
                "n_states": args.n_states,
                "destinations": dests,
                "uniform_four_class_reference": chance,
                "seed": args.seed,
                "anchors": {k: [float(x) for x in v] for k, v in anchors.items()},
                "resamples": args.resamples,
                "torch_threads": torch.get_num_threads(),
                "split_unit": "demonstration_episode",
                "device": args.device,
                "image_orientation": "rot180",
                "state_composition": list(STATE_KEYS),
                "platform": platform.platform(),
                "xgboost_version": _package_version("xgboost"),
            }.items()
        ),
        encoding="utf-8",
    )

    pooled = [t for ts in OBJECT_SOURCE_TASKS.values() for t in ts]
    by_task = [collect_states(suite_dir, [task], args.n_states, args.seed) for task in pooled]
    states = [rows[i] for i in range(args.n_states) for rows in by_task if i < len(rows)][
        : args.n_states
    ]
    print(f"states     : {len(states)}")
    # activations[site] -> list of vectors; parallel label/condition lists
    acts: dict[str, list[np.ndarray]] = {n: [] for n in site_names}
    labels: list[int] = []
    is_novel: list[bool] = []
    state_of: list[str] = []  # which state each example came from, for the group split
    executed_labels: list[int] = []

    resumed_states = 0
    resumed_seconds = 0.0
    if args.resume_from:
        previous = args.resume_from
        old_states = json.loads((previous / "states.json").read_text())
        old_config = __import__("yaml").safe_load((previous / "config.yaml").read_text())
        for key, expected in {
            "checkpoint": args.checkpoint,
            "seed": args.seed,
            "device": args.device,
            "image_orientation": "rot180",
            "state_composition": list(STATE_KEYS),
            "destinations": dests,
            "anchors": {k: [float(x) for x in v] for k, v in anchors.items()},
            "torch_threads": torch.get_num_threads(),
        }.items():
            if old_config.get(key) != expected:
                raise ValueError(f"Resume config differs at {key}")
        current_states = [{k: st[k] for k in ("task", "demo", "t")} for st in states]
        cached_path = previous / "activations.npz"
        if not cached_path.exists():
            cached_path = previous / "partial_activations.npz"
        cached = np.load(cached_path, allow_pickle=False)
        resumed_states = len(cached["y"]) // 8
        if len(cached["y"]) % 8 or old_states[:resumed_states] != current_states[:resumed_states]:
            raise ValueError("Resume cache is not an exact observation prefix")
        old_environment = json.loads((previous / "environment.json").read_text())
        if old_environment["sites"] != site_names:
            raise ValueError("Resume activation sites differ")
        for name in site_names:
            acts[name] = list(cached[name])
        labels = cached["y"].tolist()
        is_novel = cached["novel_mask"].tolist()
        state_of = cached["groups"].tolist()
        executed_labels = cached["executed_y"].tolist()
        resumed_seconds = json.loads((previous / "timing.json").read_text())["inference_seconds"]
        print(f"resumed {resumed_states} states from {previous}", flush=True)

    (out_dir / "states.json").write_text(
        json.dumps([{k: st[k] for k in ("task", "demo", "t")} for st in states], indent=2)
    )
    from huggingface_hub import try_to_load_from_cache

    model_files = {}
    for repo in (args.checkpoint, "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
        cached = try_to_load_from_cache(repo, "config.json")
        model_files[repo] = str(cached) if isinstance(cached, str) else None
    (out_dir / "environment.json").write_text(
        json.dumps(
            {
                "cached_model_configs": model_files,
                "versions": {
                    k: _package_version(k)
                    for k in (
                        "torch",
                        "numpy",
                        "scikit-learn",
                        "lerobot",
                        "transformers",
                        "huggingface_hub",
                    )
                },
                "sites": site_names,
                "activation_occurrence": 0,
                "prompts": [
                    instruction_for(obj, dest) for obj in OBJECT_SOURCE_TASKS for dest in dests
                ],
                "noise_seed": model.noise_seed,
            },
            indent=2,
        )
    )
    if args.resume_from and old_environment != json.loads(
        (out_dir / "environment.json").read_text()
    ):
        raise ValueError("Resume model revisions, software, sites, or prompts differ")
    t0 = time.time()
    for i, st in enumerate(states):
        if i < resumed_states:
            continue
        if not np.isfinite(st["ee_pos"]).all():
            raise RuntimeError(f"Non-finite end-effector position at state {i}")
        obs = {k: st[k] for k in ("agentview_rgb", "eye_in_hand_rgb")}
        obs |= {k: st[k] for k in STATE_KEYS}
        images = build_images(obs, keys, size)
        state_t = model.normalize_state(build_state(obs, sdim))
        noise = model.make_noise(1)

        for obj in OBJECT_SOURCE_TASKS:
            for di, dest in enumerate(dests):
                batch = make_batch(
                    images, state_t, instruction_for(obj, dest), model.policy, args.device
                )
                run = model.forward_with_cache(batch, sites=site_names, noise=noise)
                for n in site_names:
                    # VLM prefix pass; first denoising step for expert sites.
                    acts[n].append(pool_activation(run.occurrences(n)[0][0].float().cpu().numpy()))
                labels.append(di)
                is_novel.append(dest not in TRAINED[obj])
                state_of.append(f"{st['task']}__{st['demo']}")
                action = model.unnormalize_action(run.action)
                if not torch.isfinite(action).all():
                    raise RuntimeError(f"Non-finite action at state {i}")
                delta = action[0, :, :3].sum(dim=0).float().cpu().numpy().astype(np.float64)
                if not np.isfinite(delta).all():
                    raise RuntimeError(f"Non-finite action displacement at state {i}")
                outcome = score_direction(
                    trial_id=f"probe__{i}__{obj}__{dest}",
                    obj=obj,
                    named_destination=dest,
                    ee_pos=np.asarray(st["ee_pos"], dtype=np.float64),
                    predicted_delta=delta,
                    anchors=anchors,
                )
                executed_labels.append(
                    dests.index(outcome.chosen_destination) if outcome is not None else -1
                )
        if args.cache_activations:
            partial = out_dir / "partial_activations.tmp.npz"
            np.savez_compressed(
                partial,
                y=np.asarray(labels),
                groups=np.asarray(state_of),
                novel_mask=np.asarray(is_novel),
                executed_y=np.asarray(executed_labels),
                **{name: np.stack(acts[name]) for name in site_names},
            )
            partial.replace(out_dir / "partial_activations.npz")
            (out_dir / "timing.json").write_text(
                json.dumps(
                    {
                        "inference_seconds": resumed_seconds + time.time() - t0,
                        "states": i + 1,
                        "forwards": len(labels),
                        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        "resumed_states": resumed_states,
                    }
                )
            )
        if True:
            per = (time.time() - t0) / (i + 1 - resumed_states)
            print(
                f"    {i + 1}/{len(states)} states  ({per:.1f}s/state, "
                f"eta {per * (len(states) - i - 1) / 60:.1f} min)",
                flush=True,
            )

    (out_dir / "timing.json").write_text(
        json.dumps(
            {
                "inference_seconds": resumed_seconds + time.time() - t0,
                "states": len(states),
                "forwards": len(labels),
                "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "resumed_states": resumed_states,
            }
        )
    )
    y = np.array(labels)
    novel_mask = np.array(is_novel)
    executed_y = np.array(executed_labels, dtype=np.int64)
    print(f"\nexamples: {len(y)} total, {(~novel_mask).sum()} trained, {novel_mask.sum()} novel")

    # Train on TRAINED pairings only, test on trained-heldout AND novel.
    #
    # SPLIT BY DEMONSTRATION EPISODE, not by example. Each state contributes 8 examples
    # (2 objects x 4 destinations)
    # destinations) that share an observation and a noise draw, so an example-level split
    # puts near-identical activations in both train and test and lets the probe memorise
    # state-specific patterns. That leak showed up unmistakably: acc_trained came out at
    # 1.000 at EVERY site, including expert layers where the instruction signal is ~3e-2
    # and one where the activation is byte-identical across instructions. Grouping by
    # episode removes both repeated-state and neighboring-timestep leakage.
    groups = np.asarray(state_of)
    trained_states = np.unique(groups[~novel_mask])
    rng.shuffle(trained_states)
    cut = int(0.7 * len(trained_states))
    train_states, test_states = set(trained_states[:cut]), set(trained_states[cut:])

    tr = np.array([i for i in np.flatnonzero(~novel_mask) if groups[i] in train_states])
    te = np.array([i for i in np.flatnonzero(~novel_mask) if groups[i] in test_states])
    # Novel examples are held out by construction; also drop those whose state was trained on,
    # so no probe ever sees a test state.
    nv = np.array([i for i in np.flatnonzero(novel_mask) if groups[i] in test_states])
    if not len(tr) or not len(te) or not len(nv):
        raise RuntimeError("Too few independent episodes for a held-out split; collect more data.")
    print(f"  split by episode: {len(train_states)} train / {len(test_states)} test episodes")

    # Activations are the expensive part and are identical for every probe family, so
    # cache them once. A later probe variant then costs seconds instead of a full sweep of
    # forward passes -- which is what made the non-linear probe below cheap to add.
    if args.cache_activations:
        np.savez_compressed(
            out_dir / "activations.npz",
            y=y,
            groups=np.asarray(groups),
            novel_mask=novel_mask,
            executed_y=executed_y,
            **{s.name: np.stack(acts[s.name]) for s in sites},
        )
        print(f"  cached activations -> {out_dir / 'activations.npz'}")

    results = []
    executed_scores: dict[str, dict[str, object]] = {}
    for s in sites:
        X = np.stack(acts[s.name])
        acc_tr = fit_probe(X[tr], y[tr], X[te], y[te], seed=args.seed)
        acc_nv = fit_probe(X[tr], y[tr], X[nv], y[nv], seed=args.seed)
        y_hat_nv = predict_probe(X[tr], y[tr], X[nv], seed=args.seed)
        nv_exec = nv[executed_y[nv] >= 0]
        valid_mask = executed_y[nv] >= 0
        named_valid = (
            float(np.mean(y_hat_nv[valid_mask] == y[nv][valid_mask]))
            if len(nv_exec)
            else float("nan")
        )
        exec_acc = (
            float(np.mean(y_hat_nv[valid_mask] == executed_y[nv_exec]))
            if len(nv_exec)
            else float("nan")
        )
        paired_delta = (
            (y_hat_nv[valid_mask] == executed_y[nv_exec]).astype(float)
            - (y_hat_nv[valid_mask] == y[nv][valid_mask]).astype(float)
            if len(nv_exec)
            else np.array([], dtype=float)
        )
        delta_ci = (
            bootstrap_mean(
                paired_delta,
                resamples=args.resamples,
                seed=args.seed,
                cluster=groups[nv_exec],
            ).as_dict()
            if len(nv_exec)
            else None
        )
        y_shuf = y[tr].copy()
        rng.shuffle(y_shuf)
        acc_sh = fit_probe(X[tr], y_shuf, X[te], y[te], seed=args.seed)
        shuffled_predictions = predict_probe(X[tr], y_shuf, X[nv], seed=args.seed)
        executed_scores[s.name] = {
            "shuffled_novel_named_accuracy": float(
                np.mean(shuffled_predictions[valid_mask] == y[nv_exec])
            )
            if len(nv_exec)
            else None,
            "shuffled_novel_selected_accuracy": float(
                np.mean(shuffled_predictions[valid_mask] == executed_y[nv_exec])
            )
            if len(nv_exec)
            else None,
            "n_disagreement": int(np.sum(executed_y[nv_exec] != y[nv_exec])),
            "acc_novel_named": acc_nv,
            "acc_novel_named_on_direction_valid": named_valid,
            "acc_novel_executed": exec_acc,
            "executed_minus_named_paired_difference": delta_ci,
            "n_novel_with_direction": int(len(nv_exec)),
            "n_novel_without_direction": int(len(nv) - len(nv_exec)),
        }
        results.append(
            ProbeResult(
                site=s.name,
                layer=s.layer,
                tower=s.tower,
                n_train=len(tr),
                n_test_trained=len(te),
                n_test_novel=len(nv),
                acc_trained=acc_tr,
                acc_novel=acc_nv,
                acc_shuffled=acc_sh,
                chance=chance,
            )
        )

    nonlinear = {}
    if args.nonlinear:
        # Same split, same standardisation, same readout -- only the hypothesis class
        # differs, so any gap over the linear probe is attributable to linearity. The
        # boosted fit runs in a fresh CPU-only process because XGBoost can segfault after
        # Torch/MPS has been active on macOS/arm64.
        nonlinear = _fit_nonlinear_in_subprocess(
            out_dir, site_names, acts, y, tr, te, nv, args.seed
        )

    v = {
        "verdict": "DESCRIPTIVE",
        "reason": "Probe agreement does not establish an encoding or readout failure.",
    }
    payload: dict = {
        "verdict": v,
        "sites": [r.as_dict() for r in results],
        "execution_destination_probe": {
            "description": (
                "The same probe trained on named destinations is scored on novel examples "
                "against both the named destination and the destination selected by the "
                "predicted motion direction. This is correlational evidence only."
            ),
            "sites": executed_scores,
        },
    }
    if nonlinear:
        payload["nonlinear"] = nonlinear
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 74)
    print(f"DESTINATION PROBE  ->  {v['verdict']}")
    print("=" * 74)
    print(f"chance = {chance:.3f}\n")
    print(f"{'site':<26}{'trained':>9}{'novel':>9}{'shuffled':>10}")
    for r in sorted(results, key=lambda x: -x.acc_novel)[:15]:
        print(f"{r.site:<26}{r.acc_trained:>9.3f}{r.acc_novel:>9.3f}{r.acc_shuffled:>10.3f}")
    print(f"\n{v['reason']}")

    print("\nNOVEL TRIALS: AGREEMENT WITH NAMED VS ACTION-SELECTED DESTINATION")
    print(f"{'site':<26}{'named':>9}{'selected':>11}{'n-valid':>10}")
    for site, score in sorted(
        executed_scores.items(),
        key=lambda item: (
            -item[1]["acc_novel_executed"]
            if np.isfinite(item[1]["acc_novel_executed"])
            else float("inf")
        ),
    )[:15]:
        print(
            f"{site:<26}{score['acc_novel_named']:>9.3f}"
            f"{score['acc_novel_executed']:>11.3f}"
            f"{score['n_novel_with_direction']:>10d}"
        )

    if nonlinear:
        # The comparison that matters is linear vs boosted at the SAME site on the SAME
        # split. A boosted probe that gains only where its own shuffled control also rises
        # is fitting noise rather than finding structure, so all three are printed.
        print("\n" + "=" * 74)
        print("NON-LINEAR PROBE (gradient-boosted trees) vs LINEAR, novel pairings")
        print("=" * 74)
        print(f"{'site':<26}{'linear':>9}{'boosted':>9}{'delta':>8}{'bst-shuf':>10}")
        for r in sorted(results, key=lambda x: -nonlinear[x.site]["acc_novel"])[:15]:
            nl = nonlinear[r.site]
            print(
                f"{r.site:<26}{r.acc_novel:>9.3f}{nl['acc_novel']:>9.3f}"
                f"{nl['acc_novel'] - r.acc_novel:>+8.3f}{nl['acc_shuffled']:>10.3f}"
            )
        deltas = [nonlinear[r.site]["acc_novel"] - r.acc_novel for r in results]
        exp = [d for r, d in zip(results, deltas, strict=True) if r.tower == "expert"]
        print(
            f"\nmean(boosted - linear): all sites {np.mean(deltas):+.3f}, "
            f"expert sites {np.mean(exp):+.3f}"
        )
        print(
            f"expert sites where boosted beats linear by >0.10: "
            f"{sum(1 for d in exp if d > 0.10)}/{len(exp)}"
        )
        print("Failure of both probes does not establish that information is absent.")

    print(f"\nwritten -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
