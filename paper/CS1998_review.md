# CS1998 final review

The final inference cache contains 480 observation-command evaluations from 60 states and 30 task-demonstration episodes. Primary analysis uses 210 fitting examples, 90 held-out trained examples, and 54 valid novel examples from nine held-out episodes. No novel direction was excluded. Episode membership is disjoint across fitting and evaluation.

The initial full-run numerical audit failed on one held-out prediction. Final logistic fits use tolerance 1e-8 at every site, split, and shuffle. All eight primary-site fits now match independently optimized predictions on all 144 held-out examples per site, and their explicit einsum scores match the library decision scores. Original analyses remain under results/cs1998_full/pre_convergence_check/. convergence_correction.json records every prediction-array change. The primary paired differences are unchanged by this correction.

The fast suite passed all 73 selected tests. The 34 execution/statistics tests passed again after the final solver-convergence changes. Actual-checkpoint verification passed stock-instrumented equality, fixed-noise determinism, hook occurrence counts, exact first-state pooled-feature replay at all eight sites, and an independent motion-label calculation using the recorded four CPU threads. Earlier backend warning flags and the unrelated legacy model-loading crash are disclosed in the supplement.

The proposal and short report are two pages each. The extended writeup is six pages. All pages were rendered and visually reviewed; charts, table alignment, footers, references, and page transitions were checked. The extended writeup explicitly separates manuscript background from new course results. Three public references were verified against their arXiv records. The current project manuscript was read locally; OpenReview metadata remained behind browser verification.

The primary result does not support an expert-wide motion preference: expert L0's positive paired difference does not outperform motion controls, while later expert layers favor instruction agreement. Three of eight sites change effect sign across exploratory splits. No closed-loop behavior, causal localization, additional checkpoint, or mitigation is claimed. The source-object consistency check is labeled exploratory and does not resolve the meaning-versus-motor-plan confound.

The documents include model revision, data and package identities, prompts, split membership, uncertainty, controls, limitations, and an AI-use statement. The package includes the complete course implementation and measured cache. The report code URL points to https://github.com/agupt0318/cs1998-vla-project, which publishes the course implementation. The release archive supplies the measured activation cache for exact course reproduction.

The course guide asks the student to understand the code and choose the final plan. Student review is still required before submission. No Canvas submission was performed. The user subsequently authorized publication to the existing GitHub fork and a dedicated CS 1998 repository.

Earlier review notes in .vla_project_work/review_notes.md were inspected; no new subagents were launched for this finalization.
