Instruction or Predicted Motion?

Anant Gupta | CS 1998 final project report

1. Introduction

A readable instruction representation can coexist with an inconsistent robot action. We asked whether a fixed decoder of SmolVLA activations agrees more with the requested destination or with a destination inferred from predicted motion on unfamiliar object-destination combinations. We predicted greater motion agreement at some action-expert sites. This tests a limit of representation-based oversight in an offline instruction-following setting.

2. Related Work

SmolVLA [1] supplies the frozen policy and LIBERO [2] the manipulation observations. Our earlier project [3] studied compositional grounding failures and destination biases. The new contribution is a paired comparison of two labels for the same probe predictions, with episode-held-out evaluation. Hewitt and Liang [4] motivate controls for probe interpretation; our shuffled-label fits are a related diagnostic, rather than a replication of their control-task construction.

3. Method

We used k1000dai/smolvla_libero_finetune (revision 492ac1c5f1b7; LeRobot 0.6.0) on CPU. 60 LIBERO-Goal post-grasp states from five source tasks were crossed with bowl/wine bottle and plate/stove/cabinet top/rack (480 forward passes). Trained pairs are bowl with plate/stove/cabinet and bottle with rack/cabinet; the other three are novel. Each state's observation and Gaussian action noise (seed 0) were fixed across prompts. Mean-pooled residual-post features were cached at layers 0, 4, 8, 12 in both towers, using the first expert denoising occurrence.

The motion label maximizes cosine similarity between the summed predicted translation and directions to demonstration-derived destination anchors. It is an offline direction proxy, not observed arrival or task success. A 70/30 demonstration-episode split (seed 0) keeps neighboring states together. Standardization and logistic probes (C=0.1) were fitted only on trained pairings. Controls are held-out trained decoding, 20 label-shuffled probe fits, and a training-derived majority-motion predictor.

There were 210 training and 90 held-out trained examples; 54 valid novel trials across 9 held-out episodes (0 novel exclusions for displacement norm below 1e-6). The primary statistic is mean[1(probe=motion) - 1(probe=instruction)] on identical valid trials. Error bars use 10,000 whole-episode percentile bootstrap resamples, conditional on the fitted probe and split. All eight sites are descriptive comparisons. Split-seed sensitivity reuses the same observations.

Figure 1. Novel-trial agreement and paired differences. Error bars are episode-bootstrap 95% intervals. Expert features are from the first denoising step.

4. Results

Site | Trained | Named | Motion | Motion - named [95% CI] | Shuffle
Named / Motion

expert.L0 | 1.000 | 0.185 | 0.315 | +0.130 [+0.056, +0.204] | 0.165 / 0.287

expert.L4 | 1.000 | 0.537 | 0.222 | -0.315 [-0.463, -0.130] | 0.183 / 0.281

expert.L8 | 1.000 | 0.630 | 0.222 | -0.407 [-0.500, -0.296] | 0.175 / 0.231

expert.L12 | 1.000 | 0.481 | 0.296 | -0.185 [-0.333, -0.019] | 0.204 / 0.278

vlm.L0 | 0.367 | 0.185 | 0.315 | +0.130 [-0.019, +0.296] | 0.166 / 0.277

vlm.L4 | 0.600 | 0.389 | 0.296 | -0.093 [-0.259, +0.074] | 0.166 / 0.272

vlm.L8 | 0.789 | 0.389 | 0.241 | -0.148 [-0.259, -0.019] | 0.164 / 0.260

vlm.L12 | 0.844 | 0.370 | 0.259 | -0.111 [-0.241, +0.037] | 0.169 / 0.258

Table 1. Trained is held-out named-label decoding. Named and Motion are scores of the same probe on the same valid novel trials. Shuffle entries average 20 shuffled-label fits.

1 of eight descriptive difference intervals were wholly positive and 4 wholly negative; 1 of four expert intervals were wholly positive. Direction-proxy agreement with the command was 0.678 on valid held-out trained trials and 0.389 on valid novel trials. The majority-motion control agreed with motion labels on 0.333 of novel trials.

In exploratory splits with seeds 1, 2, 3, 4, 3 of eight sites changed the sign of their point estimate at least once. These are sensitivity checks, not independent replications. Full disagreement-subset scores, class counts, control comparisons, and split membership are in the supplement.

5. Discussion and limitations

The hypothesis of a motion preference in the action expert lacks clear support under the specified controls. Positive motion-minus-instruction scores alone are insufficient: label imbalance can favor a weak or nearly constant decoder. The majority-motion and shuffled controls are necessary context for Table 1.

The sample has one checkpoint, one benchmark suite, two object names, and a fixed noise seed. Novel labels exclude the cabinet because both cabinet pairings are trained; motion labels can still be cabinet, so a universal 25% baseline is inappropriate. Anchor calibration uses benchmark demonstrations before the split. Counterfactual commands can name an object absent from the gripper. Mean pooling and the first expert denoising step limit what is measured. Intervals exclude probe-training uncertainty; split sensitivity only partly addresses this. No closed-loop evaluation, causal intervention, mitigation, or second-checkpoint replication was performed. Failure to decode establishes no absence of information.

Compute: the pilot took 16.1 minutes. The retained 480 forward passes took 105.6 minutes of inference and checkpointing, including the reused pilot; external-service expenditure was $0. Interrupted, unsaved work is additional. Pilot memory was not recorded; full-run peak memory is in the supplement.

References and code

[1] Shukor et al. (2025). SmolVLA. https://arxiv.org/abs/2506.01844.

[2] Liu et al. (2023). LIBERO. https://arxiv.org/abs/2306.03310.

[3] Schmalz and Gupta (2026). Partial Composition, Not Blind Substitution. Existing manuscript, paper/main.tex. OpenReview browser verification prevented public metadata verification.

[4] Hewitt and Liang (2019). Designing and Interpreting Probes with Control Tasks. EMNLP. https://arxiv.org/abs/1909.03368.

Code and reproducibility artifacts: https://github.com/agupt0318/cs1998-vla-project.

AI-use statement. We used a coding agent to write the probing and evaluation scripts, and an AI assistant to brainstorm control conditions and help draft the writeup. We reviewed all code and checked every citation ourselves.
