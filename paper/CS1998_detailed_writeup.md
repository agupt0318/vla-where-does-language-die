Instruction or Predicted Motion?

Destination representations during compositional failures in SmolVLA

Anant Gupta | CS 1998 | Extended experimental writeup | October 25, 2026

Abstract

We extend an existing study of compositional instruction grounding by comparing two interpretations of the same internal destination decoder: agreement with the requested destination and agreement with a direction-based label of predicted motion. A frozen LIBERO-finetuned SmolVLA was evaluated on 60 stored states and eight commands per state, producing 480 forward passes. Logistic probes trained only on familiar object-destination pairings were evaluated on 54 novel trials from 9 held-out demonstration episodes. Layer-specific differences do not establish a general motion preference. The most positive expert-site result must be read against a majority-motion predictor and shuffled controls. Later expert sites favor instruction agreement on the primary split. The experiment gives a controlled, correlational description of a decoder and does not locate a causal grounding failure.

1. Background and safety motivation

Vision-language-action policies combine camera observations, robot state, and a language command to predict movement. Language gives the operator a way to redirect the policy, so failures on unfamiliar commands are a concrete reliability problem. Internal representations offer a possible diagnostic: if a destination can be decoded, an observer might infer that the policy understood it. That inference needs a behavioral comparison. A decoder can recover the instruction while the predicted action points elsewhere, or agree with motion for reasons unrelated to a motor plan.

SmolVLA [1] combines a vision-language backbone with a flow-matching action expert and predicts continuous action chunks. LIBERO [2] supplies manipulation tasks and stored demonstration observations. These support inexpensive frozen-model analysis, although stored-state predictions do not measure closed-loop task completion. This project studies that limited setting rather than general alignment or robot safety in deployment.

2. What the existing VLA paper establishes

The reference manuscript, Partial Composition, Not Blind Substitution [3], separates vision dominance, attention lapse, and fixed object-conditioned lookup using matched observation-command comparisons. It reports 94.6% instruction following in a visual-conflict regime, versus 74.5% in neutral states. Errors appear on untrained object-place combinations, are biased toward previously paired destinations by about ten percentage points above per-trial chance, and still depend on the commanded destination. These are earlier manuscript findings, not results newly generated for this course extension.

The manuscript also reports controlled negative mechanistic findings. A 130-site patching profile is reproduced by a no-failure control, and the action-expert probe results reverse between checkpoints. Its competence screen accepts only two of eight public LIBERO SmolVLA checkpoints. Together these findings motivate caution about both checkpoint selection and seemingly explanatory internal measurements. This extension retains the competent finetuned checkpoint and asks a narrower paired-label question; it does not claim to repair the original patching confound or test a different architecture.

3. Question, prediction, and contribution

On novel object-destination commands, does a fixed decoder agree more with the named destination or with the destination implied by predicted motion? The proposal predicts a positive motion-minus-instruction difference at some action-expert sites, especially when the labels disagree. The extension adds the motion-derived label, a paired comparison on identical held-out examples, repeated shuffled-label fits, a training-derived majority-motion baseline, and sensitivity to episode split. Hewitt and Liang [4] motivate controls for distinguishing representation content from probe learning; the shuffles here are a related diagnostic rather than their word-type control-task construction.

4. Experimental design

The checkpoint is k1000dai/smolvla_libero_finetune at revision 492ac1c5f1b7808c444fae37b75a84fdeb15e70d, using LeRobot 0.6.0. The sample contains 30 task-demonstration episodes across five source tasks, with two stored post-grasp states per episode and 12 states per task. Every state is crossed with bowl or wine bottle and plate, stove, cabinet top, or rack. Bowl-to-plate/stove/cabinet and bottle-to-rack/cabinet are trained pairings; bowl-to-rack and bottle-to-plate/stove are novel. There are 300 trained-pairing and 180 novel-pairing forward passes overall.

Both camera images use the corrected 180-degree orientation. The robot-state vector concatenates end-effector position, orientation, and gripper states. Recovered checkpoint normalization statistics are used. Each state has the same image, robot state, and fixed Gaussian action-noise draw across all eight commands. No policy parameters are trained. At residual-post layers 0, 4, 8, and 12 in each tower, features are mean-pooled over positions; expert features use the first of ten denoising occurrences.

The motion label is the anchor direction with largest cosine similarity to the sum of predicted translation vectors in the action chunk. Anchors are mean final end-effector positions in the benchmark demonstrations. A displacement norm below 1e-6 is invalid and excluded from both novel accuracy arms. This is an intended-direction proxy, not integrated dynamics, an observed endpoint, or evidence of successful grasping and placement. Exact anchors, prompts, sampled indices, and input hashes accompany the results.

5. Probe fitting and uncertainty

The primary split uses seed 0 and holds entire task-demonstration episodes together: 21 training episodes and 9 test episodes. Only trained pairings enter fitting (210 examples); novel commands from training episodes are unused. Evaluation has 90 held-out trained examples and 54 valid novel examples. There are 0 invalid novel motion labels. Standardization is fit on training features only, followed by multinomial logistic regression with C=0.1, solver tolerance 1e-8, and at most 2,000 iterations. All four named classes occur in training.

For every valid novel trial, the score is 1[probe = motion] minus 1[probe = instruction]. Its mean lies between -1 and +1. The same predictions and examples enter both accuracy arms. Percentile 95% intervals resample whole held-out episodes 10,000 times. They condition on the fitted probe and chosen split and exclude uncertainty from refitting on a new training sample. All eight site comparisons are descriptive; no site was selected as a confirmatory discovery and no multiple-testing-adjusted significance claim is made.

Controls include held-out trained decoding, 20 training-label permutations per site, and the most frequent training motion class. The same shuffled label assignments are used across sites. Shuffled ranges describe these null fits and are not calibrated episode-level confidence intervals. Seeds 1-4 change the episode split while holding the model observations and noise fixed; they test sensitivity and are not independent replications.

6. Primary results

Figure 1. Probe agreement and paired differences on 54 valid novel trials across 9 held-out episodes. Error bars are whole-episode bootstrap intervals.

| Site | Trained | Named | Motion | Motion - named [95% CI] |

| expert.L0 | 1.000 | 0.185 | 0.315 | +0.130 [+0.056, +0.204] |

| expert.L4 | 1.000 | 0.537 | 0.222 | -0.315 [-0.463, -0.130] |

| expert.L8 | 1.000 | 0.630 | 0.222 | -0.407 [-0.500, -0.296] |

| expert.L12 | 1.000 | 0.481 | 0.296 | -0.185 [-0.333, -0.019] |

| vlm.L0 | 0.367 | 0.185 | 0.315 | +0.130 [-0.019, +0.296] |

| vlm.L4 | 0.600 | 0.389 | 0.296 | -0.093 [-0.259, +0.074] |

| vlm.L8 | 0.789 | 0.389 | 0.241 | -0.148 [-0.259, -0.019] |

| vlm.L12 | 0.844 | 0.370 | 0.259 | -0.111 [-0.241, +0.037] |

The direction proxy agrees with the requested destination on 0.678 of held-out trained trials and 0.389 of novel trials. This gap describes the selected states and split; it is not a new replication of the earlier paper's matched behavioral estimate.

At expert layer 0, the paired difference is +0.130 [+0.056, +0.204]; motion agreement is 0.315. The majority-motion baseline reaches 0.333, so the positive paired difference alone is not evidence that this decoder identifies motion better than a constant predictor. At expert layer 8, the difference is -0.407 [-0.500, -0.296], favoring instruction agreement. A single narrative in which the entire expert has replaced the instruction with a motion target does not fit this primary split.

7. Class imbalance and control results

In destination order plate, stove, cabinet top, rack, training named counts are [42, 42, 84, 42]. Valid novel named counts are [18, 18, 0, 18]; motion counts are [12, 20, 18, 4]. Cabinet commands are never novel because both cabinet pairings are trained, but cabinet remains a possible motion prediction. A cabinet-only predictor therefore has zero novel named accuracy and can have a positive motion-minus-named score without reading a command. The training-selected majority class is index 2; its paired difference is +0.333 [+0.148, +0.537].

| Site | Shuffled named / motion | Motion over shuffled [95% CI] | Motion over majority [95% CI] |

| expert.L0 | 0.165 / 0.287 | +0.028 [-0.144, +0.161] | -0.019 [-0.315, +0.222] |

| expert.L4 | 0.183 / 0.281 | -0.058 [-0.183, +0.051] | -0.111 [-0.370, +0.093] |

| expert.L8 | 0.175 / 0.231 | -0.009 [-0.141, +0.112] | -0.111 [-0.389, +0.130] |

| expert.L12 | 0.204 / 0.278 | +0.019 [-0.128, +0.156] | -0.037 [-0.333, +0.204] |

| vlm.L0 | 0.166 / 0.277 | +0.038 [-0.111, +0.220] | -0.019 [-0.241, +0.204] |

| vlm.L4 | 0.166 / 0.272 | +0.024 [-0.107, +0.138] | -0.037 [-0.333, +0.204] |

| vlm.L8 | 0.164 / 0.260 | -0.019 [-0.103, +0.071] | -0.093 [-0.315, +0.093] |

| vlm.L12 | 0.169 / 0.258 | +0.001 [-0.112, +0.110] | -0.074 [-0.333, +0.148] |

A universal 25% chance line would obscure this asymmetry. A decoder that generalizes the cabinet label too often can look motion-aligned while failing on the three novel named classes. Conversely, a decoder trained on familiar pairings can be wrong about both labels. These controls constrain the interpretation of a positive difference; they do not make the motion proxy a ground-truth motor plan.

8. Disagreement trials and split sensitivity

| Site | Disagreeing trials | Named | Motion | Paired difference [95% CI] |

| expert.L0 | 33 | 0.152 | 0.364 | +0.212 [+0.077, +0.417] |

| expert.L4 | 33 | 0.606 | 0.091 | -0.515 [-0.793, -0.194] |

| expert.L8 | 33 | 0.667 | 0.000 | -0.667 [-0.844, -0.500] |

| expert.L12 | 33 | 0.485 | 0.182 | -0.303 [-0.576, -0.033] |

| vlm.L0 | 33 | 0.121 | 0.333 | +0.212 [-0.031, +0.462] |

| vlm.L4 | 33 | 0.364 | 0.212 | -0.152 [-0.429, +0.133] |

| vlm.L8 | 33 | 0.424 | 0.182 | -0.242 [-0.467, -0.030] |

| vlm.L12 | 33 | 0.364 | 0.182 | -0.182 [-0.423, +0.051] |

On trials where the labels agree, the paired difference is necessarily zero. The disagreement subset makes the competing labels explicit, but it is selected using predicted behavior and remains a descriptive subset of the same held-out observations.

| Split seed | Expert L0 | Expert L4 | Expert L8 | Expert L12 |

| 0 | +0.130 | -0.315 | -0.407 | -0.185 |

| 1 | +0.056 | -0.222 | -0.481 | -0.278 |

| 2 | +0.037 | -0.296 | -0.389 | -0.315 |

| 3 | -0.056 | -0.315 | -0.463 | -0.296 |

| 4 | -0.074 | -0.259 | -0.407 | -0.204 |

Across the four additional splits, 3 of eight sites change the sign of their point estimate at least once. Full intervals and VLM results are in the reproducibility supplement. Split-dependent conclusions should not be described as stable model mechanisms.

9. Exploratory object-consistency check

Some counterfactual commands name an object different from the source demonstration object. As a descriptive check, the table separates held-out novel commands whose object name matches the source task from those that do not. It reuses the primary probes and split and was added during final analysis. Matching the source task is not an independent verification of the grasp, and unequal destinations and task composition still confound this comparison.

| Novel subset | Trials | Expert L0 named / motion | Expert L8 named / motion |

| Source object matches command | 24 | 0.125 / 0.417 | 0.750 / 0.083 |

| Source object differs from command | 30 | 0.233 / 0.233 | 0.533 / 0.333 |

10. Validation, compute, and deviations

The retained 480 forward passes, including the reused ten-state pilot, took 105.6 minutes including per-state checkpoint writes. Recorded peak resident memory was 2.89 GiB. Inference used the local CPU with four threads and no external-service charge. Loading, verification replays, probe analyses, and discarded work are additional. Pilot memory was not measured. An interrupted earlier 17-state attempt lost its unsaved work; the retained run uses per-state atomic activation saves and validates the pilot prefix before resuming.

Actual-checkpoint verification compares stock and instrumented inference, checks fixed-noise determinism and capture invariance, counts hook occurrences, replays the first cached feature vector at all eight sites, and independently recalculates a motion label. This is a first-state check, not a replay of all 480 predictions. A one-thread replay differed slightly; replay uses the recorded four-thread setting. The unrelated legacy model-test process previously crashed while loading weights with XGBoost imported; the separate experimental-checkpoint verification avoids that import.

The initial full-run numerical audit found one of 144 held-out predictions differed between the default-tolerance logistic fit and an independent implementation. The final fits tighten the solver tolerance from 1e-4 to 1e-8 for every split and shuffle, preserving C=0.1 and all labels and features. The original outputs are retained. Numerical audit checks finite coefficients, decision scores computed by explicit einsum, saved predictions, and the independent optimized multinomial objective. This is a disclosed numerical correction rather than a new hypothesis or changed evaluation sample.

11. Interpretation and limitations

The experiment does not support a general expert-wide motion preference. It shows why a destination decoder should be compared with both the command and an action-derived label, and why that comparison needs label-distribution controls. Strong held-out decoding of familiar pairings does not guarantee compositional transfer. Layer differences and split sensitivity limit the inference to the specified decoder, checkpoint, pooling rule, and first denoising occurrence.

External validity remains limited to one small model family and one checkpoint in this extension. The original manuscript's checkpoint disagreements remain unresolved. Both towers are measured with mean-pooled activations; informative token-specific or later-denoising signals may be missed. Anchor geometry, benchmark-wide calibration that includes evaluation episodes, action-chunk cancellation, counterfactual object names, and small episode counts limit the motion proxy. The meaning-versus-motor-plan confound is not resolved by linear decoding or the exploratory object check.

A subsequent study could fit probes across denoising time, use independent geometric calibration, evaluate more competent checkpoints, and compare the proxy with closed-loop rollouts. Those experiments were not performed here. Establishing causal use would require validated interventions with matched no-failure controls; no encoding-versus-readout verdict follows from the present agreement scores.

References and reproducibility artifacts

[1] Shukor et al. (2025). SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. https://arxiv.org/abs/2506.01844.

[2] Liu et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. https://arxiv.org/abs/2306.03310.

[3] Partial Composition, Not Blind Substitution: Decomposing a Grounding Failure in Vision-Language-Action Models. Existing project manuscript, paper/main.tex. Local manuscript inspected; public OpenReview metadata was inaccessible behind browser verification.

[4] Hewitt and Liang (2019). Designing and Interpreting Probes with Control Tasks. EMNLP. https://arxiv.org/abs/1909.03368.

Code: https://github.com/agupt0318/cs1998-vla-project. The source-and-results archive contains the implementation, activation cache, exact state metadata, five split analyses, per-trial predictions, data hashes, numerical and inference checks, and environment versions. CS1998_README.md gives reproduction commands. Raw images, demonstration files, and model weights are excluded. The short report is the course-length deliverable; this extended document and CS1998_reproducibility.md provide supporting detail.

AI-use statement

We used a coding agent to write the probing and evaluation scripts, and an AI assistant to brainstorm control conditions and help draft the writeup. We reviewed all code and checked every citation ourselves.
