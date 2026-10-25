"""Create the extended, measured course writeup alongside the short report."""

from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]


def build(run: Path) -> Path:
    a = json.loads((run / "analysis_seed0.json").read_text())
    timing = json.loads((run / "timing.json").read_text())
    env = json.loads((run / "environment.json").read_text())
    d = np.load(run / "activations.npz", allow_pickle=False)
    p = np.load(run / "predictions_seed0.npz", allow_pickle=False)
    sites = sorted(a["sites"], key=lambda x: (x.split(".")[0], int(x.split(".")[1][1:])))
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyDetail", fontName="Helvetica", fontSize=10, leading=14, spaceAfter=8
        )
    )
    styles.add(ParagraphStyle(name="CaptionDetail", fontSize=8.5, leading=11, spaceAfter=8))
    styles["Heading1"].fontSize = 17
    styles["Heading1"].leading = 21
    styles["Heading2"].keepWithNext = True
    story = []
    md = []

    def para(s, style="BodyDetail"):
        story.append(Paragraph(escape(s), styles[style]))
        md.append(s + "\n")

    def heading(s):
        para(s, "Heading2")

    def table(rows, widths):
        wrapped = [
            [Paragraph(escape(str(c)), styles["CaptionDetail"]) for c in row] for row in rows
        ]
        t = Table(wrapped, colWidths=widths, repeatRows=1, hAlign="LEFT")
        t.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf0f5")),
                    ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f5f7f9")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.extend([t, Spacer(1, 10)])
        md.extend(["| " + " | ".join(map(str, row)) + " |\n" for row in rows])

    def ci(x):
        return f"{x['value']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]"

    para("Instruction or Predicted Motion?", "Title")
    para("Destination representations during compositional failures in SmolVLA", "Heading1")
    para("Anant Gupta | CS 1998 | Extended experimental writeup | October 25, 2026")
    heading("Abstract")
    para(
        f"We extend an existing study of compositional instruction grounding by comparing two interpretations of the same internal destination decoder: agreement with the requested destination and agreement with a direction-based label of predicted motion. A frozen LIBERO-finetuned SmolVLA was evaluated on 60 stored states and eight commands per state, producing 480 forward passes. Logistic probes trained only on familiar object-destination pairings were evaluated on {a['n_valid_novel']} novel trials from {a['n_test_episodes']} held-out demonstration episodes. Layer-specific differences do not establish a general motion preference. The most positive expert-site result must be read against a majority-motion predictor and shuffled controls. Later expert sites favor instruction agreement on the primary split. The experiment gives a controlled, correlational description of a decoder and does not locate a causal grounding failure."
    )
    heading("1. Background and safety motivation")
    para(
        "Vision-language-action policies combine camera observations, robot state, and a language command to predict movement. Language gives the operator a way to redirect the policy, so failures on unfamiliar commands are a concrete reliability problem. Internal representations offer a possible diagnostic: if a destination can be decoded, an observer might infer that the policy understood it. That inference needs a behavioral comparison. A decoder can recover the instruction while the predicted action points elsewhere, or agree with motion for reasons unrelated to a motor plan."
    )
    para(
        "SmolVLA [1] combines a vision-language backbone with a flow-matching action expert and predicts continuous action chunks. LIBERO [2] supplies manipulation tasks and stored demonstration observations. These support inexpensive frozen-model analysis, although stored-state predictions do not measure closed-loop task completion. This project studies that limited setting rather than general alignment or robot safety in deployment."
    )
    heading("2. What the existing VLA paper establishes")
    para(
        "The reference manuscript, Partial Composition, Not Blind Substitution [3], separates vision dominance, attention lapse, and fixed object-conditioned lookup using matched observation-command comparisons. It reports 94.6% instruction following in a visual-conflict regime, versus 74.5% in neutral states. Errors appear on untrained object-place combinations, are biased toward previously paired destinations by about ten percentage points above per-trial chance, and still depend on the commanded destination. These are earlier manuscript findings, not results newly generated for this course extension."
    )
    para(
        "The manuscript also reports controlled negative mechanistic findings. A 130-site patching profile is reproduced by a no-failure control, and the action-expert probe results reverse between checkpoints. Its competence screen accepts only two of eight public LIBERO SmolVLA checkpoints. Together these findings motivate caution about both checkpoint selection and seemingly explanatory internal measurements. This extension retains the competent finetuned checkpoint and asks a narrower paired-label question; it does not claim to repair the original patching confound or test a different architecture."
    )
    heading("3. Question, prediction, and contribution")
    para(
        "On novel object-destination commands, does a fixed decoder agree more with the named destination or with the destination implied by predicted motion? The proposal predicts a positive motion-minus-instruction difference at some action-expert sites, especially when the labels disagree. The extension adds the motion-derived label, a paired comparison on identical held-out examples, repeated shuffled-label fits, a training-derived majority-motion baseline, and sensitivity to episode split. Hewitt and Liang [4] motivate controls for distinguishing representation content from probe learning; the shuffles here are a related diagnostic rather than their word-type control-task construction."
    )
    heading("4. Experimental design")
    revision = Path(env["cached_model_configs"]["k1000dai/smolvla_libero_finetune"]).parent.name
    para(
        f"The checkpoint is k1000dai/smolvla_libero_finetune at revision {revision}, using LeRobot {env['versions']['lerobot']}. The sample contains 30 task-demonstration episodes across five source tasks, with two stored post-grasp states per episode and 12 states per task. Every state is crossed with bowl or wine bottle and plate, stove, cabinet top, or rack. Bowl-to-plate/stove/cabinet and bottle-to-rack/cabinet are trained pairings; bowl-to-rack and bottle-to-plate/stove are novel. There are 300 trained-pairing and 180 novel-pairing forward passes overall."
    )
    para(
        "Both camera images use the corrected 180-degree orientation. The robot-state vector concatenates end-effector position, orientation, and gripper states. Recovered checkpoint normalization statistics are used. Each state has the same image, robot state, and fixed Gaussian action-noise draw across all eight commands. No policy parameters are trained. At residual-post layers 0, 4, 8, and 12 in each tower, features are mean-pooled over positions; expert features use the first of ten denoising occurrences."
    )
    para(
        "The motion label is the anchor direction with largest cosine similarity to the sum of predicted translation vectors in the action chunk. Anchors are mean final end-effector positions in the benchmark demonstrations. A displacement norm below 1e-6 is invalid and excluded from both novel accuracy arms. This is an intended-direction proxy, not integrated dynamics, an observed endpoint, or evidence of successful grasping and placement. Exact anchors, prompts, sampled indices, and input hashes accompany the results."
    )
    heading("5. Probe fitting and uncertainty")
    para(
        f"The primary split uses seed 0 and holds entire task-demonstration episodes together: {a['n_train_episodes']} training episodes and {a['n_test_episodes']} test episodes. Only trained pairings enter fitting ({a['n_train']} examples); novel commands from training episodes are unused. Evaluation has {a['n_test_trained']} held-out trained examples and {a['n_valid_novel']} valid novel examples. There are {a['n_invalid_novel']} invalid novel motion labels. Standardization is fit on training features only, followed by multinomial logistic regression with C=0.1, solver tolerance 1e-8, and at most 2,000 iterations. All four named classes occur in training."
    )
    para(
        "For every valid novel trial, the score is 1[probe = motion] minus 1[probe = instruction]. Its mean lies between -1 and +1. The same predictions and examples enter both accuracy arms. Percentile 95% intervals resample whole held-out episodes 10,000 times. They condition on the fitted probe and chosen split and exclude uncertainty from refitting on a new training sample. All eight site comparisons are descriptive; no site was selected as a confirmatory discovery and no multiple-testing-adjusted significance claim is made."
    )
    para(
        "Controls include held-out trained decoding, 20 training-label permutations per site, and the most frequent training motion class. The same shuffled label assignments are used across sites. Shuffled ranges describe these null fits and are not calibrated episode-level confidence intervals. Seeds 1-4 change the episode split while holding the model observations and noise fixed; they test sensitivity and are not independent replications."
    )
    heading("6. Primary results")
    story.append(Image(str(run / "execution_probe.png"), width=510, height=217))
    para(
        f"Figure 1. Probe agreement and paired differences on {a['n_valid_novel']} valid novel trials across {a['n_test_episodes']} held-out episodes. Error bars are whole-episode bootstrap intervals.",
        "CaptionDetail",
    )
    rows = [["Site", "Trained", "Named", "Motion", "Motion - named [95% CI]"]]
    for n in sites:
        v = a["sites"][n]
        rows.append(
            [
                n.replace(".resid_post", ""),
                f"{v['trained_accuracy']:.3f}",
                f"{v['named']['value']:.3f}",
                f"{v['selected']['value']:.3f}",
                ci(v["delta"]),
            ]
        )
    table(rows, [83, 55, 55, 55, 262])
    para(
        f"The direction proxy agrees with the requested destination on {a['behavior_trained_accuracy']:.3f} of held-out trained trials and {a['behavior_novel_accuracy']:.3f} of novel trials. This gap describes the selected states and split; it is not a new replication of the earlier paper's matched behavioral estimate."
    )
    v = a["sites"]["expert.L0.resid_post"]
    later = a["sites"]["expert.L8.resid_post"]
    para(
        f"At expert layer 0, the paired difference is {ci(v['delta'])}; motion agreement is {v['selected']['value']:.3f}. The majority-motion baseline reaches {a['majority_selected_accuracy']:.3f}, so the positive paired difference alone is not evidence that this decoder identifies motion better than a constant predictor. At expert layer 8, the difference is {ci(later['delta'])}, favoring instruction agreement. A single narrative in which the entire expert has replaced the instruction with a motion target does not fit this primary split."
    )
    heading("7. Class imbalance and control results")
    para(
        f"In destination order plate, stove, cabinet top, rack, training named counts are {a['all_training_named_counts']}. Valid novel named counts are {a['novel_named_counts']}; motion counts are {a['novel_selected_counts']}. Cabinet commands are never novel because both cabinet pairings are trained, but cabinet remains a possible motion prediction. A cabinet-only predictor therefore has zero novel named accuracy and can have a positive motion-minus-named score without reading a command. The training-selected majority class is index {a['majority_selected_class_from_training']}; its paired difference is {ci(a['majority_paired_difference'])}."
    )
    rows = [
        [
            "Site",
            "Shuffled named / motion",
            "Motion over shuffled [95% CI]",
            "Motion over majority [95% CI]",
        ]
    ]
    for n in sites:
        v = a["sites"][n]
        rows.append(
            [
                n.replace(".resid_post", ""),
                f"{v['shuffled_named_mean']:.3f} / {v['shuffled_selected_mean']:.3f}",
                ci(v["selected_over_shuffled"]),
                ci(v["selected_over_majority"]),
            ]
        )
    table(rows, [83, 103, 162, 162])
    para(
        "A universal 25% chance line would obscure this asymmetry. A decoder that generalizes the cabinet label too often can look motion-aligned while failing on the three novel named classes. Conversely, a decoder trained on familiar pairings can be wrong about both labels. These controls constrain the interpretation of a positive difference; they do not make the motion proxy a ground-truth motor plan."
    )
    heading("8. Disagreement trials and split sensitivity")
    rows = [["Site", "Disagreeing trials", "Named", "Motion", "Paired difference [95% CI]"]]
    for n in sites:
        v = a["sites"][n]
        rows.append(
            [
                n.replace(".resid_post", ""),
                v["n_disagreement"],
                f"{v['disagreement_named']['value']:.3f}",
                f"{v['disagreement_selected']['value']:.3f}",
                ci(v["disagreement_delta"]),
            ]
        )
    table(rows, [83, 75, 50, 50, 252])
    para(
        "On trials where the labels agree, the paired difference is necessarily zero. The disagreement subset makes the competing labels explicit, but it is selected using predicted behavior and remains a descriptive subset of the same held-out observations."
    )
    rows = [["Split seed", "Expert L0", "Expert L4", "Expert L8", "Expert L12"]]
    all_a = []
    for file in sorted(run.glob("analysis_seed*.json")):
        z = json.loads(file.read_text())
        all_a.append(z)
        rows.append(
            [z["split_seed"]]
            + [
                f"{z['sites'][f'expert.L{layer}.resid_post']['delta']['value']:+.3f}"
                for layer in [0, 4, 8, 12]
            ]
        )
    table(rows, [70, 110, 110, 110, 110])
    changed = sum(
        any(
            np.sign(z["sites"][n]["delta"]["value"]) != np.sign(a["sites"][n]["delta"]["value"])
            for z in all_a[1:]
        )
        for n in sites
    )
    para(
        f"Across the four additional splits, {changed} of eight sites change the sign of their point estimate at least once. Full intervals and VLM results are in the reproducibility supplement. Split-dependent conclusions should not be described as stable model mechanisms."
    )
    heading("9. Exploratory object-consistency check")
    para(
        "Some counterfactual commands name an object different from the source demonstration object. As a descriptive check, the table separates held-out novel commands whose object name matches the source task from those that do not. It reuses the primary probes and split and was added during final analysis. Matching the source task is not an independent verification of the grasp, and unequal destinations and task composition still confound this comparison."
    )
    ix = p["test_novel_indices"]
    cmd_bottle = (ix % 8) // 4 == 1
    source_bottle = np.array(["wine_bottle" in str(g) for g in d["groups"][ix]])
    rows = [["Novel subset", "Trials", "Expert L0 named / motion", "Expert L8 named / motion"]]
    strat = []
    for label, mask in [
        ("Source object matches command", cmd_bottle == source_bottle),
        ("Source object differs from command", cmd_bottle != source_bottle),
    ]:
        row = [label, int(mask.sum())]
        for site in ["expert.L0.resid_post", "expert.L8.resid_post"]:
            pred = p[site][mask]
            row.append(
                f"{np.mean(pred == p['named'][mask]):.3f} / {np.mean(pred == p['selected'][mask]):.3f}"
            )
        rows.append(row)
        strat.append(row)
    table(rows, [170, 40, 150, 150])
    (run / "exploratory_object_consistency.json").write_text(json.dumps(strat, indent=2))
    heading("10. Validation, compute, and deviations")
    para(
        f"The retained 480 forward passes, including the reused ten-state pilot, took {timing['inference_seconds'] / 60:.1f} minutes including per-state checkpoint writes. Recorded peak resident memory was {timing['peak_rss_bytes'] / 1024**3:.2f} GiB. Inference used the local CPU with four threads and no external-service charge. Loading, verification replays, probe analyses, and discarded work are additional. Pilot memory was not measured. An interrupted earlier 17-state attempt lost its unsaved work; the retained run uses per-state atomic activation saves and validates the pilot prefix before resuming."
    )
    para(
        "Actual-checkpoint verification compares stock and instrumented inference, checks fixed-noise determinism and capture invariance, counts hook occurrences, replays the first cached feature vector at all eight sites, and independently recalculates a motion label. This is a first-state check, not a replay of all 480 predictions. A one-thread replay differed slightly; replay uses the recorded four-thread setting. The unrelated legacy model-test process previously crashed while loading weights with XGBoost imported; the separate experimental-checkpoint verification avoids that import."
    )
    para(
        "The initial full-run numerical audit found one of 144 held-out predictions differed between the default-tolerance logistic fit and an independent implementation. The final fits tighten the solver tolerance from 1e-4 to 1e-8 for every split and shuffle, preserving C=0.1 and all labels and features. The original outputs are retained. Numerical audit checks finite coefficients, decision scores computed by explicit einsum, saved predictions, and the independent optimized multinomial objective. This is a disclosed numerical correction rather than a new hypothesis or changed evaluation sample."
    )
    heading("11. Interpretation and limitations")
    para(
        "The experiment does not support a general expert-wide motion preference. It shows why a destination decoder should be compared with both the command and an action-derived label, and why that comparison needs label-distribution controls. Strong held-out decoding of familiar pairings does not guarantee compositional transfer. Layer differences and split sensitivity limit the inference to the specified decoder, checkpoint, pooling rule, and first denoising occurrence."
    )
    para(
        "External validity remains limited to one small model family and one checkpoint in this extension. The original manuscript's checkpoint disagreements remain unresolved. Both towers are measured with mean-pooled activations; informative token-specific or later-denoising signals may be missed. Anchor geometry, benchmark-wide calibration that includes evaluation episodes, action-chunk cancellation, counterfactual object names, and small episode counts limit the motion proxy. The meaning-versus-motor-plan confound is not resolved by linear decoding or the exploratory object check."
    )
    para(
        "A subsequent study could fit probes across denoising time, use independent geometric calibration, evaluate more competent checkpoints, and compare the proxy with closed-loop rollouts. Those experiments were not performed here. Establishing causal use would require validated interventions with matched no-failure controls; no encoding-versus-readout verdict follows from the present agreement scores."
    )
    heading("References and reproducibility artifacts")
    for ref in [
        "[1] Shukor et al. (2025). SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. https://arxiv.org/abs/2506.01844.",
        "[2] Liu et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. https://arxiv.org/abs/2306.03310.",
        "[3] Partial Composition, Not Blind Substitution: Decomposing a Grounding Failure in Vision-Language-Action Models. Existing project manuscript, paper/main.tex. Local manuscript inspected; public OpenReview metadata was inaccessible behind browser verification.",
        "[4] Hewitt and Liang (2019). Designing and Interpreting Probes with Control Tasks. EMNLP. https://arxiv.org/abs/1909.03368.",
    ]:
        para(ref, "CaptionDetail")
    para(
        "Code: https://github.com/agupt0318/cs1998-vla-project. The source-and-results archive contains the implementation, activation cache, exact state metadata, five split analyses, per-trial predictions, data hashes, numerical and inference checks, and environment versions. CS1998_README.md gives reproduction commands. Raw images, demonstration files, and model weights are excluded. The short report is the course-length deliverable; this extended document and CS1998_reproducibility.md provide supporting detail."
    )
    heading("AI-use statement")
    para(
        "We used a coding agent to write the probing and evaluation scripts, and an AI assistant to brainstorm control conditions and help draft the writeup. We reviewed all code and checked every citation ourselves."
    )

    def footer(c, doc):
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.HexColor("#506070"))
        c.drawString(45, 24, "CS 1998 | Extended experimental writeup")
        c.drawRightString(567, 24, str(doc.page))

    target = ROOT / "paper/CS1998_detailed_writeup.pdf"
    SimpleDocTemplate(
        str(target),
        pagesize=(612, 792),
        leftMargin=45,
        rightMargin=45,
        topMargin=38,
        bottomMargin=40,
        title="Instruction or Predicted Motion? Extended writeup",
        author="Anant Gupta",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / "paper/CS1998_detailed_writeup.md").write_text("\n".join(md))
    print(target)
    return target


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    build(parser.parse_args().run)
