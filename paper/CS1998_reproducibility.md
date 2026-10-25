# CS1998 reproducibility supplement

This supplement accompanies the two-page final report. The original pilot is a runtime and pipeline check, with only two held-out episodes. Final interpretation uses the full run.

## Exact model and software

```json
{
  "cached_model_configs": {
    "k1000dai/smolvla_libero_finetune": "/Users/agupta/.cache/huggingface/hub/models--k1000dai--smolvla_libero_finetune/snapshots/492ac1c5f1b7808c444fae37b75a84fdeb15e70d/config.json",
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct": "/Users/agupta/.cache/huggingface/hub/models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct/snapshots/7b375e1b73b11138ff12fe22c8f2822d8fe03467/config.json"
  },
  "versions": {
    "torch": "2.11.0",
    "numpy": "2.2.6",
    "scikit-learn": "1.9.0",
    "lerobot": "0.6.0",
    "transformers": "5.5.4",
    "huggingface_hub": "1.28.0"
  },
  "sites": [
    "vlm.L0.resid_post",
    "vlm.L4.resid_post",
    "vlm.L8.resid_post",
    "vlm.L12.resid_post",
    "expert.L0.resid_post",
    "expert.L4.resid_post",
    "expert.L8.resid_post",
    "expert.L12.resid_post"
  ],
  "activation_occurrence": 0,
  "prompts": [
    "put the bowl on the plate",
    "put the bowl on the stove",
    "put the bowl on top of the cabinet",
    "put the bowl on the rack",
    "put the wine bottle on the plate",
    "put the wine bottle on the stove",
    "put the wine bottle on top of the cabinet",
    "put the wine bottle on the rack"
  ],
  "noise_seed": 0
}
```

Full installed package versions are in requirements_cs1998.lock.txt. Absolute cached model paths identify the exact snapshot revisions; personal filesystem prefixes are not needed for reproduction.

## Sampling, labels, and geometry

60 states across 30 task+demo episodes; two states per episode, 12 states per source task. All eight commands are evaluated on every state. Training pairings are bowl→plate/stove/cabinet and wine bottle→rack/cabinet; the remaining three pairings are novel. No cabinet-named novel examples exist.

```yaml
run_id: "cs1998_full"
experiment: "layerwise_execution_destination_probe"
checkpoint: "k1000dai/smolvla_libero_finetune"
n_states: 60
destinations: ["the plate", "the stove", "top of the cabinet", "the rack"]
uniform_four_class_reference: 0.25
seed: 0
anchors: {"the plate": [0.05177321591101755, 0.02598617121908347, 0.968549932205958], "the stove": [-0.23250660959318906, 0.2560819095096693, 0.9933318521404848], "top of the cabinet": [-0.012820469148772188, -0.19122408159163057, 1.22735947248791], "the rack": [-0.1853942575492872, -0.25755863478749297, 1.2055436912208006]}
resamples: 10000
torch_threads: 4
split_unit: "demonstration_episode"
device: "cpu"
image_orientation: "rot180"
state_composition: ["ee_pos", "ee_ori", "gripper_states"]
platform: "macOS-27.0-arm64-arm-64bit"
xgboost_version: "3.4.1"
```

State metadata is in states.json. Inputs use both camera images rotated 180 degrees, resized using the existing build_images helper, and state concatenation [ee_pos, ee_ori, gripper_states]. Normalization buffers are recovered from the checkpoint. Anchors are mean final end-effector positions from benchmark demonstrations, calculated before the probe split; calibration includes test episodes. Thus motion-label calibration is benchmark-dependent and is not an out-of-sample geometric generalization test.

## Split and counts

Split seed 0: 21 training episodes, 9 valid held-out episodes; 210 trained-pairing examples for fitting, 90 held-out trained examples, and 54 held-out novel examples. Novel exclusions: 0; held-out trained exclusions for behavior scores: 0. Novel examples from training episodes are unused.

Destination order: the plate, the stove, top of the cabinet, the rack. Training named counts: [42, 42, 84, 42]; held-out trained named counts: [18, 18, 36, 18]; valid novel named counts: [18, 18, 0, 18]; valid novel motion counts: [12, 20, 18, 4].

Training episodes:

- put_the_bowl_on_the_plate__demo_18
- put_the_bowl_on_the_plate__demo_23
- put_the_bowl_on_the_plate__demo_3
- put_the_bowl_on_the_plate__demo_36
- put_the_bowl_on_the_stove__demo_18
- put_the_bowl_on_the_stove__demo_2
- put_the_bowl_on_the_stove__demo_23
- put_the_bowl_on_the_stove__demo_36
- put_the_bowl_on_the_stove__demo_4
- put_the_bowl_on_top_of_the_cabinet__demo_18
- put_the_bowl_on_top_of_the_cabinet__demo_2
- put_the_bowl_on_top_of_the_cabinet__demo_36
- put_the_wine_bottle_on_the_rack__demo_18
- put_the_wine_bottle_on_the_rack__demo_2
- put_the_wine_bottle_on_the_rack__demo_23
- put_the_wine_bottle_on_the_rack__demo_3
- put_the_wine_bottle_on_the_rack__demo_4
- put_the_wine_bottle_on_top_of_the_cabinet__demo_2
- put_the_wine_bottle_on_top_of_the_cabinet__demo_23
- put_the_wine_bottle_on_top_of_the_cabinet__demo_36
- put_the_wine_bottle_on_top_of_the_cabinet__demo_4

Held-out episodes:

- put_the_bowl_on_the_plate__demo_2
- put_the_bowl_on_the_plate__demo_4
- put_the_bowl_on_the_stove__demo_3
- put_the_bowl_on_top_of_the_cabinet__demo_23
- put_the_bowl_on_top_of_the_cabinet__demo_3
- put_the_bowl_on_top_of_the_cabinet__demo_4
- put_the_wine_bottle_on_the_rack__demo_36
- put_the_wine_bottle_on_top_of_the_cabinet__demo_18
- put_the_wine_bottle_on_top_of_the_cabinet__demo_3

## All-site controls

The majority motion class is top of the cabinet, chosen on trained-pairing examples in training episodes only. Its motion accuracy is 0.333; its paired difference is +0.333 [+0.148, +0.537].

| Site | Trained / shuffled trained | Novel named / shuffled named | Novel motion / shuffled motion | Motion minus named [95% CI] | Motion over majority [95% CI] |
|---|---|---|---|---|---|
| expert.L0.resid_post | 1.000 / 0.288 | 0.185 / 0.165 | 0.315 / 0.287 | +0.130 [+0.056, +0.204] | -0.019 [-0.315, +0.222] |
| expert.L4.resid_post | 1.000 / 0.281 | 0.537 / 0.183 | 0.222 / 0.281 | -0.315 [-0.463, -0.130] | -0.111 [-0.370, +0.093] |
| expert.L8.resid_post | 1.000 / 0.288 | 0.630 / 0.175 | 0.222 / 0.231 | -0.407 [-0.500, -0.296] | -0.111 [-0.389, +0.130] |
| expert.L12.resid_post | 1.000 / 0.299 | 0.481 / 0.204 | 0.296 / 0.278 | -0.185 [-0.333, -0.019] | -0.037 [-0.333, +0.204] |
| vlm.L0.resid_post | 0.367 / 0.297 | 0.185 / 0.166 | 0.315 / 0.277 | +0.130 [-0.019, +0.296] | -0.019 [-0.241, +0.204] |
| vlm.L4.resid_post | 0.600 / 0.297 | 0.389 / 0.166 | 0.296 / 0.272 | -0.093 [-0.259, +0.074] | -0.037 [-0.333, +0.204] |
| vlm.L8.resid_post | 0.789 / 0.295 | 0.389 / 0.164 | 0.241 / 0.260 | -0.148 [-0.259, -0.019] | -0.093 [-0.315, +0.093] |
| vlm.L12.resid_post | 0.844 / 0.296 | 0.370 / 0.169 | 0.259 / 0.258 | -0.111 [-0.241, +0.037] | -0.074 [-0.333, +0.148] |

## Disagreement subset

| Site | Trials / episodes | Named agreement | Motion agreement | Paired difference [95% CI] |
|---|---|---|---|---|
| expert.L0.resid_post | 33 / 9 | +0.152 [+0.037, +0.263] | +0.364 [+0.211, +0.552] | +0.212 [+0.077, +0.417] |
| expert.L4.resid_post | 33 / 9 | +0.606 [+0.439, +0.793] | +0.091 [+0.000, +0.265] | -0.515 [-0.793, -0.194] |
| expert.L8.resid_post | 33 / 9 | +0.667 [+0.500, +0.844] | +0.000 [+0.000, +0.000] | -0.667 [-0.844, -0.500] |
| expert.L12.resid_post | 33 / 9 | +0.485 [+0.344, +0.677] | +0.182 [+0.046, +0.346] | -0.303 [-0.576, -0.033] |
| vlm.L0.resid_post | 33 / 9 | +0.121 [+0.036, +0.194] | +0.333 [+0.114, +0.548] | +0.212 [-0.031, +0.462] |
| vlm.L4.resid_post | 33 / 9 | +0.364 [+0.212, +0.522] | +0.212 [+0.081, +0.355] | -0.152 [-0.429, +0.133] |
| vlm.L8.resid_post | 33 / 9 | +0.424 [+0.289, +0.568] | +0.182 [+0.077, +0.270] | -0.242 [-0.467, -0.030] |
| vlm.L12.resid_post | 33 / 9 | +0.364 [+0.200, +0.536] | +0.182 [+0.077, +0.270] | -0.182 [-0.423, +0.051] |

## Split sensitivity

Seeds 1-4 change the demonstration split while retaining the same activations, input noise, model, hyperparameters, and analysis. These are exploratory sensitivity analyses. The confidence intervals condition on each fitted probe and exclude its fitting uncertainty. Twenty trial-label shuffles per site are descriptive controls, not a calibrated multiple-testing procedure. Their 2.5%/97.5% range is a range across fitted null probes, not an episode confidence interval.

| Split seed | Valid novel trials / held-out episodes | Expert L0 | Expert L4 | Expert L8 | Expert L12 | VLM L0 | VLM L4 | VLM L8 | VLM L12 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 54 / 9 | +0.130 [+0.056, +0.204] | -0.315 [-0.463, -0.130] | -0.407 [-0.500, -0.296] | -0.185 [-0.333, -0.019] | +0.130 [-0.019, +0.296] | -0.093 [-0.259, +0.074] | -0.148 [-0.259, -0.019] | -0.111 [-0.241, +0.037] |
| 1 | 54 / 9 | +0.056 [-0.056, +0.167] | -0.222 [-0.389, -0.037] | -0.481 [-0.611, -0.315] | -0.278 [-0.463, -0.074] | +0.370 [+0.204, +0.537] | +0.056 [-0.093, +0.204] | +0.037 [-0.148, +0.241] | -0.111 [-0.333, +0.148] |
| 2 | 54 / 9 | +0.037 [-0.093, +0.130] | -0.296 [-0.370, -0.204] | -0.389 [-0.481, -0.278] | -0.315 [-0.426, -0.204] | +0.185 [+0.074, +0.333] | -0.019 [-0.111, +0.093] | -0.074 [-0.185, +0.037] | -0.130 [-0.167, -0.074] |
| 3 | 54 / 9 | -0.056 [-0.185, +0.074] | -0.315 [-0.500, -0.111] | -0.463 [-0.574, -0.315] | -0.296 [-0.370, -0.222] | +0.222 [+0.074, +0.389] | -0.222 [-0.444, +0.000] | -0.167 [-0.333, +0.019] | -0.167 [-0.370, +0.093] |
| 4 | 54 / 9 | -0.074 [-0.185, +0.037] | -0.259 [-0.407, -0.074] | -0.407 [-0.519, -0.278] | -0.204 [-0.352, -0.037] | +0.278 [+0.130, +0.444] | -0.074 [-0.185, +0.056] | -0.204 [-0.259, -0.130] | -0.204 [-0.278, -0.111] |

## Compute, deviations, and validation

```json
{
  "inference_seconds": 6335.839439868927,
  "states": 60,
  "forwards": 480,
  "peak_rss_bytes": 3101147136,
  "resumed_states": 10
}
```

Inference timing sums retained pilot and resumed-run work, including per-state cache writes. An earlier full-run attempt was interrupted after 17 states without a saved cache, so about 28 minutes of computation were discarded. Model loading, analysis, verification replays, and document generation are additional. Peak RSS is in bytes on macOS. Peak pilot memory was not captured, which is a deviation from the original plan; full-run peak RSS is recorded. External-service charges were $0.

The full cache reuses the ten-state pilot prefix after exact state and environment checks. Future resumes require a distinct output directory and validate checkpoint, seed, device, image orientation, destination order, anchors, state composition, thread count, prompts, software, and snapshot metadata. Data-file hashes are recorded separately.

The existing model-test process with XGBoost imported crashed during multithreaded Transformers weight loading. A separate process without XGBoost verifies the actual experimental checkpoint against upstream inference, capture invariance, fixed-noise determinism, occurrence counts, cache replay, and independent direction scoring. Its measured results are in inference_verification.json.

NumPy uses Apple Accelerate in this environment and emitted floating-point warnings on finite matrix products. The initial full-run audit also found one prediction disagreement between the default-tolerance probe fit and an independently optimized logistic objective. Final course fits use solver tolerance 1e-8, including every shuffled control and split; C=0.1 and the scientific design are unchanged. Initial analyses are preserved in pre_convergence_check/. numerical_audit.json checks finite values, explicit einsum scores, saved predictions, and independently optimized predictions. Warnings alone do not validate a result.

Independent reviews and the final requirements audit are in paper/CS1998_review.md. The instructor-facing report does not claim that the student has already read or understood every source line; student review is still required by the course guide.

## Reproduction

See CS1998_README.md for setup and exact commands. Reanalysis from the supplied activation cache avoids downloading weights or demonstrations. trials.csv lists every observation-command combination; test predictions are blank for unused or fitting examples, preserving the held-out design. Cached activations and predictions use NumPy NPZ without pickle. Raw camera observations and checkpoint weights are not redistributed.
