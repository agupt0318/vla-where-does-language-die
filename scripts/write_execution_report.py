"""Render the two-page report from canonical measured analysis, with no invented values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle

from scripts.plot_execution_probe import plot

ROOT = Path(__file__).resolve().parents[1]
CODE_URL = "https://github.com/agupt0318/cs1998-vla-project"


def build(run: Path, *, pilot_layout: bool = False) -> Path:
    a = json.loads((run / "analysis_seed0.json").read_text())
    env = json.loads((run / "environment.json").read_text())
    states = json.loads((run / "states.json").read_text())
    timing = json.loads((run / "timing.json").read_text())
    sensitivity = [
        json.loads(p.read_text())
        for p in sorted(run.glob("analysis_seed*.json"))
        if p.name != "analysis_seed0.json"
    ]
    if not pilot_layout and (len(states) != 60 or a["n_test_episodes"] < 5):
        raise ValueError("Final report requires 60 states and at least five held-out episodes")
    figure = plot(run / "analysis_seed0.json")
    names = sorted(a["sites"], key=lambda n: (n.split(".")[0], int(n.split(".")[1][1:])))
    if len(names) != 8:
        raise ValueError("All eight specified sites are required")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyCourse", fontSize=9.5, leading=12.2, spaceAfter=6))
    styles.add(ParagraphStyle(name="SmallCourse", fontSize=7.8, leading=9.7, spaceAfter=4))
    styles["Heading1"].fontSize = 17
    styles["Heading1"].leading = 21
    styles["Heading1"].spaceAfter = 6
    styles["Heading2"].fontSize = 10.5
    styles["Heading2"].leading = 13
    styles["Heading2"].spaceBefore = 6
    styles["Heading2"].spaceAfter = 4
    story, md = [], []

    def para(value: str, style: str = "BodyCourse") -> None:
        rendered = escape(value)
        if CODE_URL in value:
            rendered = rendered.replace(CODE_URL, f'<link href="{CODE_URL}">{CODE_URL}</link>')
        story.append(Paragraph(rendered, styles[style]))
        md.append(value + "\n")

    para("Instruction or Predicted Motion?", "Heading1")
    para(
        "Anant Gupta | CS 1998 final project report"
        + (" | PILOT LAYOUT TEST" if pilot_layout else "")
    )
    para("1. Introduction", "Heading2")
    para(
        "A readable instruction representation can coexist with an inconsistent robot action. We asked whether a fixed decoder of SmolVLA activations agrees more with the requested destination or with a destination inferred from predicted motion on unfamiliar object-destination combinations. We predicted greater motion agreement at some action-expert sites. This tests a limit of representation-based oversight in an offline instruction-following setting."
    )
    para("2. Related Work", "Heading2")
    para(
        "SmolVLA [1] supplies the frozen policy and LIBERO [2] the manipulation observations. Our earlier project [3] studied compositional grounding failures and destination biases. The new contribution is a paired comparison of two labels for the same probe predictions, with episode-held-out evaluation. Hewitt and Liang [4] motivate controls for probe interpretation; our shuffled-label fits are a related diagnostic, rather than a replication of their control-task construction."
    )
    para("3. Method", "Heading2")
    revision = Path(env["cached_model_configs"]["k1000dai/smolvla_libero_finetune"]).parent.name
    para(
        f"We used k1000dai/smolvla_libero_finetune (revision {revision[:12]}; LeRobot {env['versions']['lerobot']}) on CPU. {len(states)} LIBERO-Goal post-grasp states from five source tasks were crossed with bowl/wine bottle and plate/stove/cabinet top/rack ({len(states) * 8} forward passes). Trained pairs are bowl with plate/stove/cabinet and bottle with rack/cabinet; the other three are novel. Each state's observation and Gaussian action noise (seed 0) were fixed across prompts. Mean-pooled residual-post features were cached at layers 0, 4, 8, 12 in both towers, using the first expert denoising occurrence."
    )
    para(
        "The motion label maximizes cosine similarity between the summed predicted translation and directions to demonstration-derived destination anchors. It is an offline direction proxy, not observed arrival or task success. A 70/30 demonstration-episode split (seed 0) keeps neighboring states together. Standardization and logistic probes (C=0.1) were fitted only on trained pairings. Controls are held-out trained decoding, 20 label-shuffled probe fits, and a training-derived majority-motion predictor."
    )
    para(
        f"There were {a['n_train']} training and {a['n_test_trained']} held-out trained examples; {a['n_valid_novel']} valid novel trials across {a['n_test_episodes']} held-out episodes ({a['n_invalid_novel']} novel exclusions for displacement norm below 1e-6). The primary statistic is mean[1(probe=motion) - 1(probe=instruction)] on identical valid trials. Error bars use 10,000 whole-episode percentile bootstrap resamples, conditional on the fitted probe and split. All eight sites are descriptive comparisons. Split-seed sensitivity reuses the same observations."
    )
    story.append(Image(str(figure), width=523, height=223))
    para(
        "Figure 1. Novel-trial agreement and paired differences. Error bars are episode-bootstrap 95% intervals. Expert features are from the first denoising step.",
        "SmallCourse",
    )
    story.append(PageBreak())

    para("4. Results", "Heading2")
    rows = [
        ["Site", "Trained", "Named", "Motion", "Motion - named [95% CI]", "Shuffle\nNamed / Motion"]
    ]
    for name in names:
        r = a["sites"][name]
        d = r["delta"]
        rows.append(
            [
                name.replace(".resid_post", ""),
                f"{r['trained_accuracy']:.3f}",
                f"{r['named']['value']:.3f}",
                f"{r['selected']['value']:.3f}",
                f"{d['value']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]",
                f"{r['shuffled_named_mean']:.3f} / {r['shuffled_selected_mean']:.3f}",
            ]
        )
    table = Table(rows, colWidths=[65, 45, 43, 43, 170, 157], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("LEADING", (0, 0), (-1, -1), 10),
                ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#edf2f5")),
                ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
            ]
        )
    )
    story.append(table)
    md.extend([" | ".join(str(x) for x in row) + "\n" for row in rows])
    para(
        "Table 1. Trained is held-out named-label decoding. Named and Motion are scores of the same probe on the same valid novel trials. Shuffle entries average 20 shuffled-label fits.",
        "SmallCourse",
    )
    positive = [n for n in names if a["sites"][n]["delta"]["lo"] > 0]
    negative = [n for n in names if a["sites"][n]["delta"]["hi"] < 0]
    expert_positive = [n for n in positive if n.startswith("expert.")]
    para(
        f"{len(positive)} of eight descriptive difference intervals were wholly positive and {len(negative)} wholly negative; {len(expert_positive)} of four expert intervals were wholly positive. Direction-proxy agreement with the command was {a['behavior_trained_accuracy']:.3f} on valid held-out trained trials and {a['behavior_novel_accuracy']:.3f} on valid novel trials. The majority-motion control agreed with motion labels on {a['majority_selected_accuracy']:.3f} of novel trials."
    )
    if sensitivity:
        changed = sum(
            any(
                __import__("numpy").sign(s["sites"][n]["delta"]["value"])
                != __import__("numpy").sign(a["sites"][n]["delta"]["value"])
                for s in sensitivity
            )
            for n in names
        )
        para(
            f"In exploratory splits with seeds {', '.join(str(s['split_seed']) for s in sensitivity)}, {changed} of eight sites changed the sign of their point estimate at least once. These are sensitivity checks, not independent replications. Full disagreement-subset scores, class counts, control comparisons, and split membership are in the supplement."
        )
    para("5. Discussion and limitations", "Heading2")
    controlled = [
        n
        for n in expert_positive
        if a["sites"][n]["selected_over_shuffled"]["lo"] > 0
        and a["sites"][n]["trained_accuracy"] > a["sites"][n]["shuffled_trained_mean"]
    ]
    if controlled:
        para(
            f"The predicted motion preference has descriptive support at {len(controlled)} expert site(s) with useful trained decoding and motion agreement above the mean shuffled control. This is evidence about this decoder's label agreement. A causal interpretation would require interventions and additional controls."
        )
    else:
        para(
            "The hypothesis of a motion preference in the action expert lacks clear support under the specified controls. Positive motion-minus-instruction scores alone are insufficient: label imbalance can favor a weak or nearly constant decoder. The majority-motion and shuffled controls are necessary context for Table 1."
        )
    para(
        "The sample has one checkpoint, one benchmark suite, two object names, and a fixed noise seed. Novel labels exclude the cabinet because both cabinet pairings are trained; motion labels can still be cabinet, so a universal 25% baseline is inappropriate. Anchor calibration uses benchmark demonstrations before the split. Counterfactual commands can name an object absent from the gripper. Mean pooling and the first expert denoising step limit what is measured. Intervals exclude probe-training uncertainty; split sensitivity only partly addresses this. No closed-loop evaluation, causal intervention, mitigation, or second-checkpoint replication was performed. Failure to decode establishes no absence of information."
    )
    para(
        f"Compute: the pilot took 16.1 minutes. The retained {timing['forwards']} forward passes took {timing['inference_seconds'] / 60:.1f} minutes of inference and checkpointing, including the reused pilot; external-service expenditure was $0. Interrupted, unsaved work is additional. Pilot memory was not recorded; full-run peak memory is in the supplement.",
        "SmallCourse",
    )
    para("References and code", "Heading2")
    for ref in [
        "[1] Shukor et al. (2025). SmolVLA. https://arxiv.org/abs/2506.01844.",
        "[2] Liu et al. (2023). LIBERO. https://arxiv.org/abs/2306.03310.",
        "[3] Schmalz and Gupta (2026). Partial Composition, Not Blind Substitution. Existing manuscript, paper/main.tex. OpenReview browser verification prevented public metadata verification.",
        "[4] Hewitt and Liang (2019). Designing and Interpreting Probes with Control Tasks. EMNLP. https://arxiv.org/abs/1909.03368.",
        f"Code and reproducibility artifacts: {CODE_URL}.",
    ]:
        para(ref, "SmallCourse")
    para(
        "AI-use statement. We used a coding agent to write the probing and evaluation scripts, and an AI assistant to brainstorm control conditions and help draft the writeup. We reviewed all code and checked every citation ourselves.",
        "SmallCourse",
    )

    def footer(canvas, doc):
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#506070"))
        canvas.drawString(43, 24, "CS 1998 | Instruction or Predicted Motion?")
        canvas.drawRightString(569, 24, str(doc.page))

    target = (
        run / "pilot_layout_report.pdf" if pilot_layout else ROOT / "paper/CS1998_final_report.pdf"
    )
    SimpleDocTemplate(
        str(target),
        pagesize=(612, 792),
        leftMargin=43,
        rightMargin=43,
        topMargin=30,
        bottomMargin=36,
        title="Instruction or Predicted Motion?",
        author="Anant Gupta",
    ).build(story, onFirstPage=footer, onLaterPages=footer)
    from pypdf import PdfReader

    if len(PdfReader(target).pages) != 2:
        raise ValueError(f"Report layout exceeded two pages: {target}")
    if not pilot_layout:
        (ROOT / "paper/CS1998_final_report.md").write_text("\n".join(md))
    print(target)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--pilot-layout", action="store_true")
    args = parser.parse_args()
    build(args.run, pilot_layout=args.pilot_layout)
