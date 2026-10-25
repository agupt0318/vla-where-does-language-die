# CS1998: Instruction or Predicted Motion?

Course repository: https://github.com/agupt0318/cs1998-vla-project. The reports and code link refer to this repository.

This extension compares a fixed destination decoder with two labels on the same novel-composition trials: the named destination and a destination proxy inferred from predicted motion. It makes correlational claims only. The earlier VLA project and its legacy causal-verdict helpers are separate from this analysis.

## Deliverables

- `paper/CS1998_project_proposal.pdf`: revised two-page proposal.
- `paper/CS1998_final_report.pdf`: two-page report generated from measured results.
- `paper/CS1998_detailed_writeup.pdf`: extended background, methods, results, controls, and limitations.
- `paper/CS1998_reproducibility.md`: exact prompts, versions, class counts, controls, and split sensitivity.
- `paper/CS1998_review.md`: independent reviews, corrections, and final requirements audit.
- `results/cs1998_full/`: cached features, labels, predictions, analysis, timing, and numerical checks. Camera observations and checkpoint weights are not redistributed.

## Reanalysis without model weights or demonstrations

Use Python 3.12. Install the analysis packages, or install the recorded environment from `requirements_cs1998.lock.txt`. The full lock describes the measured macOS environment; platform-specific wheels may require a Linux equivalent.

```bash
python -m venv .venv
.venv/bin/python -m pip install numpy scipy scikit-learn matplotlib reportlab pypdf pyyaml
```

The GitHub release provides `CS1998_source_and_results.zip`, including the measured activation cache. Download and extract it, then use its `CS1998_project/` directory (or copy `results/cs1998_full/` into this checkout). Run from the repository root:

```bash
.venv/bin/python -m scripts.analyze_execution_probe results/cs1998_full/activations.npz --seed 0 --shuffles 20 --resamples 10000
.venv/bin/python -m scripts.analyze_execution_probe results/cs1998_full/activations.npz --seed 1 --shuffles 20 --resamples 10000
.venv/bin/python -m scripts.analyze_execution_probe results/cs1998_full/activations.npz --seed 2 --shuffles 20 --resamples 10000
.venv/bin/python -m scripts.analyze_execution_probe results/cs1998_full/activations.npz --seed 3 --shuffles 20 --resamples 10000
.venv/bin/python -m scripts.analyze_execution_probe results/cs1998_full/activations.npz --seed 4 --shuffles 20 --resamples 10000
.venv/bin/python -m scripts.audit_execution_numerics results/cs1998_full
.venv/bin/python -m scripts.write_execution_report results/cs1998_full
.venv/bin/python -m scripts.write_course_supplement results/cs1998_full
.venv/bin/python -m scripts.write_course_detailed results/cs1998_full
```

The fixed logistic probe has C=0.1, training-only standardization, solver tolerance 1e-8, and maximum 2,000 solver iterations. All four named destinations must be represented in training. Splits use task+demonstration identifiers, so adjacent observations stay together. Novel examples in training episodes are unused. Both accuracy arms exclude invalid motion directions. The primary result is a paired difference with whole-episode bootstrap intervals, not two separately bootstrapped accuracies.

## Fresh inference

The original repository setup is in README.md. Install the model runtime in a fresh environment:

```bash
.venv/bin/python -m pip install -e '.[vla,dev]'
.venv/bin/python -m pip install reportlab pypdf
.venv/bin/python scripts/download_data.py
```

The exact checkpoint snapshot is `k1000dai/smolvla_libero_finetune@492ac1c5f1b7808c444fae37b75a84fdeb15e70d`. The VLM snapshot is `HuggingFaceTB/SmolVLM2-500M-Video-Instruct@7b375e1b73b11138ff12fe22c8f2822d8fe03467`. Download those snapshots before an offline run, and verify the `environment.json` snapshot paths match them. The downloader uses the public `yifengzhu-hf/LIBERO-datasets` repository; final `data_sha256.json` records the exact local input bytes. Subsequent changes in public data or checkpoints require rechecking those identities.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u scripts/run_probe.py --checkpoint k1000dai/smolvla_libero_finetune --device cpu --n-states 10 --seed 0 --resamples 10000 --cache-activations --run-id course_pilot_new
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python -u scripts/run_probe.py --checkpoint k1000dai/smolvla_libero_finetune --device cpu --n-states 60 --seed 0 --resamples 10000 --cache-activations --run-id course_full_new --resume-from results/course_pilot_new
```

Use a new output run-id for every fresh or resumed run. `partial_activations.npz` is saved atomically after each complete state. To recover an interrupted run, pass its directory as `--resume-from` and choose another output directory. Completed caches are protected from overwrite. Resume checks include the state prefix, checkpoint, seed, device, image orientation, destination order, anchor values, state composition, thread count, prompts, package versions, and snapshot metadata. Do not change the input demonstration files between runs.

The 10-state pilot took 16.1 minutes locally. A 60-state run is roughly 97 minutes before analysis; exact timing is in the output. Local execution has no external-service charge.

## Checks

```bash
.venv/bin/python -m pytest -m 'not slow and not sim' -q
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python scripts/verify_execution_inference.py results/cs1998_full
```

The exact thread count matters for bitwise cache replay on this CPU backend. A one-thread replay had small feature differences; the four-thread replay matched all eight pooled activations exactly. The standalone checkpoint check also verifies upstream parity, deterministic fixed-noise actions, activation occurrence counts, and an independently calculated motion label. It uses no simulator.

Apple Accelerate emits floating-point warnings on some finite matrix products in this environment. `audit_execution_numerics.py` checks finite values, recomputes decision scores using explicit einsum, and independently optimizes the multinomial logistic objective without matrix multiplication. All audited held-out predictions must match the saved analysis. The unrelated legacy model pytest process crashed during Transformers weight loading with XGBoost imported; the standalone actual-checkpoint verification is recorded separately.

## Interpretation and course requirements

A positive motion-minus-named difference can be explained by class imbalance or a nearly constant decoder. Read it together with trained-pairing performance, shuffled-label controls, and the majority-motion baseline. The cabinet appears in motion labels but has no novel named examples. Intervals are conditional on the fitted probe and split, and site comparisons are descriptive. Five split seeds reuse the same observations and are not independent replications.

The course permits coding agents but requires the student to read and understand the code and check the results. Read the report and supplement before submitting the proposal PDF, report PDF, and code link on Canvas. This workflow does not submit anything to Canvas.
