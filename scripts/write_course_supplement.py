"""Write exact reproducibility details, all-site controls, and human-readable trial records."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def build(run: Path) -> Path:
    a = json.loads((run / "analysis_seed0.json").read_text())
    config = yaml.safe_load((run / "config.yaml").read_text())
    env = json.loads((run / "environment.json").read_text())
    states = json.loads((run / "states.json").read_text())
    timing = json.loads((run / "timing.json").read_text())
    d = np.load(run / "activations.npz", allow_pickle=False)
    p = np.load(run / "predictions_seed0.npz", allow_pickle=False)
    destinations = config["destinations"]
    sites = sorted(a["sites"], key=lambda n: (n.split(".")[0], int(n.split(".")[1][1:])))
    novel_indices = p["test_novel_indices"]
    trained_indices = p["test_trained_indices"]
    training = set(p["train_indices"].tolist())
    trial_rows = []
    for i in range(len(d["y"])):
        state = states[i // 8]
        obj = ("bowl", "wine bottle")[(i % 8) // 4]
        yi = int(d["y"][i])
        selected = int(d["executed_y"][i])
        row = {
            "trial_index": i,
            "task": state["task"],
            "demo": state["demo"],
            "t": state["t"],
            "object": obj,
            "prompt": env["prompts"][i % 8],
            "named_destination": destinations[yi],
            "motion_destination": destinations[selected] if selected >= 0 else "INVALID",
            "condition": "novel" if d["novel_mask"][i] else "trained",
            "split": "train"
            if i in training
            else (
                "unused_train_episode_novel" if d["groups"][i] in a["train_episodes"] else "test"
            ),
        }
        for site in sites:
            novel_match = np.flatnonzero(novel_indices == i)
            trained_match = np.flatnonzero(trained_indices == i)
            pred = (
                int(p[site][novel_match[0]])
                if len(novel_match)
                else int(p[site + "__trained"][trained_match[0]])
                if len(trained_match)
                else -1
            )
            row[site] = destinations[pred] if pred >= 0 else ""
        trial_rows.append(row)
    with (run / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(trial_rows[0]))
        writer.writeheader()
        writer.writerows(trial_rows)

    def ci(est):
        return (
            "empty subset"
            if est is None
            else f"{est['value']:+.3f} [{est['lo']:+.3f}, {est['hi']:+.3f}]"
        )

    text = [
        "# CS1998 reproducibility supplement\n",
        "This supplement accompanies the two-page final report. The original pilot is a runtime and pipeline check, with only two held-out episodes. Final interpretation uses the full run.\n",
        "## Exact model and software\n",
        "```json\n" + json.dumps(env, indent=2) + "\n```\n",
        "Full installed package versions are in requirements_cs1998.lock.txt. Absolute cached model paths identify the exact snapshot revisions; personal filesystem prefixes are not needed for reproduction.\n",
        "## Sampling, labels, and geometry\n",
        f"{len(states)} states across {len(np.unique(d['groups']))} task+demo episodes; two states per episode, 12 states per source task. All eight commands are evaluated on every state. Training pairings are bowl→plate/stove/cabinet and wine bottle→rack/cabinet; the remaining three pairings are novel. No cabinet-named novel examples exist.\n",
        "```yaml\n" + (run / "config.yaml").read_text() + "\n```\n",
        "State metadata is in states.json. Inputs use both camera images rotated 180 degrees, resized using the existing build_images helper, and state concatenation [ee_pos, ee_ori, gripper_states]. Normalization buffers are recovered from the checkpoint. Anchors are mean final end-effector positions from benchmark demonstrations, calculated before the probe split; calibration includes test episodes. Thus motion-label calibration is benchmark-dependent and is not an out-of-sample geometric generalization test.\n",
        "## Split and counts\n",
        f"Split seed 0: {a['n_train_episodes']} training episodes, {a['n_test_episodes']} valid held-out episodes; {a['n_train']} trained-pairing examples for fitting, {a['n_test_trained']} held-out trained examples, and {a['n_test_novel']} held-out novel examples. Novel exclusions: {a['n_invalid_novel']}; held-out trained exclusions for behavior scores: {a['n_invalid_test_trained']}. Novel examples from training episodes are unused.\n",
        f"Destination order: {', '.join(destinations)}. Training named counts: {a['all_training_named_counts']}; held-out trained named counts: {a['heldout_trained_named_counts']}; valid novel named counts: {a['novel_named_counts']}; valid novel motion counts: {a['novel_selected_counts']}.\n",
        "Training episodes:\n\n" + "\n".join("- " + g for g in a["train_episodes"]) + "\n",
        "Held-out episodes:\n\n" + "\n".join("- " + g for g in a["test_episodes"]) + "\n",
        "## All-site controls\n",
        f"The majority motion class is {destinations[a['majority_selected_class_from_training']]}, chosen on trained-pairing examples in training episodes only. Its motion accuracy is {a['majority_selected_accuracy']:.3f}; its paired difference is {ci(a['majority_paired_difference'])}.\n",
        "| Site | Trained / shuffled trained | Novel named / shuffled named | Novel motion / shuffled motion | Motion minus named [95% CI] | Motion over majority [95% CI] |\n|---|---|---|---|---|---|",
    ]
    for site in sites:
        r = a["sites"][site]
        text.append(
            f"| {site} | {r['trained_accuracy']:.3f} / {r['shuffled_trained_mean']:.3f} | {r['named']['value']:.3f} / {r['shuffled_named_mean']:.3f} | {r['selected']['value']:.3f} / {r['shuffled_selected_mean']:.3f} | {ci(r['delta'])} | {ci(r['selected_over_majority'])} |"
        )
    text += [
        "\n## Disagreement subset\n",
        "| Site | Trials / episodes | Named agreement | Motion agreement | Paired difference [95% CI] |\n|---|---|---|---|---|",
    ]
    for site in sites:
        r = a["sites"][site]
        text.append(
            f"| {site} | {r['n_disagreement']} / {r['n_disagreement_episodes']} | {ci(r['disagreement_named'])} | {ci(r['disagreement_selected'])} | {ci(r['disagreement_delta'])} |"
        )
    text += [
        "\n## Split sensitivity\n",
        "Seeds 1-4 change the demonstration split while retaining the same activations, input noise, model, hyperparameters, and analysis. These are exploratory sensitivity analyses. The confidence intervals condition on each fitted probe and exclude its fitting uncertainty. Twenty trial-label shuffles per site are descriptive controls, not a calibrated multiple-testing procedure. Their 2.5%/97.5% range is a range across fitted null probes, not an episode confidence interval.\n",
        "| Split seed | Valid novel trials / held-out episodes | Expert L0 | Expert L4 | Expert L8 | Expert L12 | VLM L0 | VLM L4 | VLM L8 | VLM L12 |\n|---|---|---|---|---|---|---|---|---|---|",
    ]
    for path in sorted(run.glob("analysis_seed*.json")):
        s = json.loads(path.read_text())
        text.append(
            f"| {s['split_seed']} | {s['n_valid_novel']} / {s['n_test_episodes']} | "
            + " | ".join(ci(s["sites"][site]["delta"]) for site in sites)
            + " |"
        )
    text += [
        "\n## Compute, deviations, and validation\n",
        "```json\n" + json.dumps(timing, indent=2) + "\n```\n",
        "Inference timing sums retained pilot and resumed-run work, including per-state cache writes. An earlier full-run attempt was interrupted after 17 states without a saved cache, so about 28 minutes of computation were discarded. Model loading, analysis, verification replays, and document generation are additional. Peak RSS is in bytes on macOS. Peak pilot memory was not captured, which is a deviation from the original plan; full-run peak RSS is recorded. External-service charges were $0.\n",
        "The full cache reuses the ten-state pilot prefix after exact state and environment checks. Future resumes require a distinct output directory and validate checkpoint, seed, device, image orientation, destination order, anchors, state composition, thread count, prompts, software, and snapshot metadata. Data-file hashes are recorded separately.\n",
        "The existing model-test process with XGBoost imported crashed during multithreaded Transformers weight loading. A separate process without XGBoost verifies the actual experimental checkpoint against upstream inference, capture invariance, fixed-noise determinism, occurrence counts, cache replay, and independent direction scoring. Its measured results are in inference_verification.json.\n",
        "NumPy uses Apple Accelerate in this environment and emitted floating-point warnings on finite matrix products. The initial full-run audit also found one prediction disagreement between the default-tolerance probe fit and an independently optimized logistic objective. Final course fits use solver tolerance 1e-8, including every shuffled control and split; C=0.1 and the scientific design are unchanged. Initial analyses are preserved in pre_convergence_check/. numerical_audit.json checks finite values, explicit einsum scores, saved predictions, and independently optimized predictions. Warnings alone do not validate a result.\n",
        "Independent reviews and the final requirements audit are in paper/CS1998_review.md. The instructor-facing report does not claim that the student has already read or understood every source line; student review is still required by the course guide.\n",
        "## Reproduction\n",
        "See CS1998_README.md for setup and exact commands. Reanalysis from the supplied activation cache avoids downloading weights or demonstrations. trials.csv lists every observation-command combination; test predictions are blank for unused or fitting examples, preserving the held-out design. Cached activations and predictions use NumPy NPZ without pickle. Raw camera observations and checkpoint weights are not redistributed.\n",
    ]
    target = ROOT / "paper/CS1998_reproducibility.md"
    target.write_text("\n".join(text))
    print(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    build(parser.parse_args().run)
