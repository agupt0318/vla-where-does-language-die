Instruction or Predicted Motion?

Destination representations during compositional failures in SmolVLA

Anant Gupta | CS 1998: Introduction to AI Safety & Alignment | Proposal

1. Introduction

A robot policy can have a readable representation of an instruction while predicting motion toward a different destination. Representation-based oversight therefore needs a distinction between what a decoder reads and what the policy predicts. On unfamiliar object-destination combinations, does a fixed probe of SmolVLA activations agree more with the requested destination or with a destination inferred from the predicted action chunk?

I predict greater agreement with the motion-derived destination at some action-expert sites, especially when instruction and motion labels disagree. The course contribution is a new paired label comparison and its controls, extending my existing VLA project. This is a study of instruction-following reliability and diagnostic limits; every result will have a correlational interpretation.

2. Related Work

Model and observations. SmolVLA [1] is a compact vision-language-action policy, and LIBERO [2] supplies manipulation demonstrations. Together they permit a frozen-model study on local hardware.

Prior findings and probe controls. Our manuscript [3] finds strong instruction following under visual conflict, but degraded generalization to novel object-place pairings. Its action-expert probe results reverse between checkpoints. This extension compares instruction and motion labels for the same decoder on the same trials. Hewitt and Liang [4] motivate probe controls. Repeated shuffled-label fits and a majority-motion baseline test weak decoding and label imbalance; they do not establish causal use.

3. Method

Model and sample. Use k1000dai/smolvla_libero_finetune through the existing wrapper and LeRobot 0.6.0. Sample 60 LIBERO-Goal states across five source tasks and cross each with bowl/wine bottle and plate/stove/cabinet top/rack (480 forward passes). Trained pairings are bowl with plate, stove, or cabinet, and bottle with rack or cabinet; the other three are novel. Source-task and demonstration identifiers define episodes. Keep each episode in one split. Record checkpoint revisions, versions, state indices, prompts, and seed 0.

Features and motion labels. Keep the observation and Gaussian action-noise draw fixed across instructions. Cache mean-pooled residual-post activations at layers 0, 4, 8, and 12 in both the VLM and action expert, using the first expert denoising occurrence. Assign the motion label by maximum cosine similarity between the summed predicted translation and directions to demonstration-derived destination anchors. This is an offline direction proxy. Exclude displacement norms below 1e-6 from both accuracies and report the count.

3. Method (continued)

Split and controls. Use an approximately 70/30 episode split. Train standardized logistic probes (C=0.1) on named destinations from trained pairings only, with scaling fitted on training data. Evaluate held-out trained pairings as a decoding sanity check and novel pairings as the target. Fit 20 label-shuffled probes on the same features and split, and score each against both test labels. Include a majority-motion baseline chosen using training episodes only. Retain all four destination classes in training; never reuse training episodes to enlarge the test set.

Primary metric. On each valid novel trial, subtract the indicator that the probe matches the named destination from the indicator that it matches the motion-derived destination. Report the mean paired difference and both accuracies at all eight specified sites. Compute 95% percentile intervals using 10,000 resamples of entire held-out episodes. Report sample and episode counts, class frequencies, and the subset where the labels disagree. The intervals are conditional on the fitted probe and split. Additional split seeds are exploratory sensitivity checks on the same observations.

Interpretation. A positive difference with useful held-out trained decoding and performance above shuffled controls supports greater agreement with predicted motion. A negative difference supports greater agreement with the instruction. An interval spanning zero is inconclusive. A positive difference that a majority-motion baseline also explains is weak evidence for a motion-specific representation. Poor controls or missing training classes prevent interpretation. Require at least five held-out episodes for substantive conclusions; the pilot is a runtime check. Site comparisons are descriptive; selecting the largest effect is exploratory. Decoding cannot establish absence of information or a causal failure mechanism.

Compute and fallback. Use the existing local CPU environment for frozen inference and CPU probe fitting, with external-service cost $0. The ten-state pilot completed 80 forward passes in 16.1 minutes, suggesting about 97 minutes for 60 states before analysis. Record peak memory in the full run; it was not captured in the pilot. Save intermediate activations after each state. If resources are insufficient, reduce the state count, retain all eight sites, and disclose the resulting uncertainty. A second checkpoint is an optional extension.

Limitations and output. Destination geometry, unequal class frequencies, counterfactual commands for an object absent from the gripper, and correlated demonstration states constrain interpretation. Produce a 1-2 page technical report with a labeled figure, uncertainty intervals, limitations, references, a code link, and an AI-use statement. Include a reproducibility supplement with the exact prompts, splits, predictions, and environment. This revision includes the measured pilot runtime; the hypothesis and primary comparison are unchanged.

References

[1] Shukor et al. (2025). SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics. https://arxiv.org/abs/2506.01844.

[2] Liu et al. (2023). LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. https://arxiv.org/abs/2306.03310.

[3] Schmalz and Gupta (2026). Partial Composition, Not Blind Substitution: Decomposing a Grounding Failure in Vision-Language-Action Models. Existing project manuscript, paper/main.tex. The linked OpenReview page requires browser verification: https://openreview.net/forum?id=DPOBsubW55.

[4] Hewitt and Liang (2019). Designing and Interpreting Probes with Control Tasks. EMNLP. https://arxiv.org/abs/1909.03368.

AI-use statement

We used a coding agent to write the probing and evaluation scripts, and an AI assistant to brainstorm control conditions and help draft the writeup. We reviewed all code and checked every citation ourselves.
