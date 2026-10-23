# Third night: localization/classification cascade and specialist ensemble

User approved testing all three proposed schemes on 22 October 2026.
Registered before any new scientific training. Deadline **2026-10-23T08:00:00+03:00**.
No subagents, publication, paid services, commits, PRs or deployment. Preserve all
earlier results, source-package files and frozen registrations. Never reopen test.

## Fixed data, controls and budget

Reuse second-night FIT/calibration/outer manifests, three folds, seeds 20261022 and
20261023; 204 original development frames, 31 groups, original test excluded.
Student_6 precedence unchanged. FIT ignores excluded, as in the existing control.
Existing six cosine30 models/results are the COMMON CONTROL, reused with verified
hashes and exactly matching FIT/seed/settings, not selected best seeds. No new
control fitting is necessary. Their predictions and calibration remain immutable.

New budget: **12 pipeline jobs, 24 component fits**. Each fold/seed has:

1. cascade: one class-agnostic detector + one eight-way crop classifier (background
   plus seven classes). Six pipelines = six detector + six classifier fits.
2. specialists: one binary filamentous detector + one six-class detector for the
   remaining classes. Six pipelines = twelve detector fits.

All 18 new detectors: identical cosine30 recipe/COCO initialization/resize/BN/LR
to second-night cosine30. Remap FIT labels explicitly; background frames for an
individual specialist remain in FIT. Do NOT change prepared annotations in place.
Class-agnostic detector learns only the seven target groups, not flocs/debris or
out-of-scope microorganisms. Full frame resolution 1280x960, min score .05,
within-detector NMS .5, max 100 detections. Final pipeline cap is also 100/frame.

## Independent crop classifier, fixed recipe

ResNet50 initialized from the same local COCO detector's backbone.body tensors,
not released microorganism weights. No external download. ImageNet input
normalization as used by the detector. Fine-tune layer4 and new FC only; all BN
statistics frozen. AdamW LR 1e-4, weight decay 1e-4, cosine decay to 0.1 LR,
20 epochs, batch32, seed matching the detector. Final epoch only, no best-epoch
selection. Square-root inverse crop-label-frequency sampler, capped at3, includes
background, N replacement draws/epoch. Resize crops preserving aspect ratio by
padding to square, then 224x224; context padding 10% of bbox dimensions each side.
Quarter-turn/flip augmentation on FIT crops only, deterministic per epoch/index.

Training crops ONLY from usable FIT frames: each GT box once + one deterministic
jitter (edge perturbation <=10%, accept only IoU>=.7). Add predicted FIT proposals:
positive if maximum IoU>=.5 with a target (use that target class), negative if max
IoU<.1 with ALL targets; discard ambiguous interval. Positive proposals capped
two per GT, negative proposals capped five per frame. Exclude predictions inside
ignore via using only ignore-free FIT frames. Persist crop manifest and provenance.
Negatives are annotation-relative: incomplete labels can make them false negatives;
mining on a detector's own FIT is not out-of-fold hard-negative mining. Report this
limitation rather than claiming unbiased negative coverage. No cal/outer crops fit.

Cascade inference: one winning eight-way class per proposed box; background wins
are discarded. Final score = detector objectness * winning class probability;
class-wise NMS .5 after classification, max100. Objectness floor .05 is fixed;
do not tune detector recall thresholds on outer validation.

## Specialists and merging

Filamentous specialist labels only filamentous_bacteria as target; other organisms
are background for that specialist. Other specialist uses six contiguous labels.
Primary merge `specialists`: greedy cross-specialist duplicate suppression at
IoU>.8, higher raw score wins, deterministic tie order; within-specialist overlaps
unchanged. Scores across specialists are not guaranteed calibrated; true organisms
can overlap. Predeclared diagnostic `specialists_union` concatenates without this
cross-specialist suppression. Both use the same top100 cap and are reported; do not
choose a merging rule from outer outcomes. No seven-model per-class ensemble in
this budget: it is not one of the three user-approved proposed comparisons.

## Calibration, evaluation and decision

Reuse second-night fixed_020, global_f1, and PRIMARY count_guarded selection
functions unchanged. Calibrate each pipeline/merge variant on its calibration
frames, write frozen JSON BEFORE any outer inference. Class support fallback and
grid .05:.05:.95 unchanged. Original test remains excluded, no test evaluator.

For all methods report per-fold and pooled OOF separately per seed, then mean/SD
across two seeds: class/macro F1, precision/recall, per-class count MAE and bias
with/without ignore filtering. Do not merge duplicated images across seeds.
Diagnostics at fixed pre-calibration .05: class-agnostic one-to-one localization
precision/recall at IoU .5, false positives and duplicate overlaps; matched-box
classification accuracy; cascade oracle GT-crop classification confusion/accuracy
on outer validation (diagnostic only, not a deployable score or selection input).
Persist inference durations, proposal counts and merge suppression counts.

Primary candidates cascade and specialists compared against cosine30 control.
Eligibility unchanged: mean macroF1 >= control-.01; attached AND filamentous MAE
each <=90% control; no classF1 drops >.02. specialists_union is diagnostic only.
Report all failures and incomplete pairs; no blind retries, adaptive recipes,
threshold grids, new folds, or further models after the budget.

The outer folds were already inspected and motivated this study. This is adaptive
development, NOT an independent or fully nested generalization test. Control reuse
does not change this. Pipeline compute/parameter budgets are higher than control;
report accuracy/cost tradeoff, not architecture-only causality. Classifier sees
limited rare-class support. No superiority-to-release claim (release train IDs
unknown), no automatic replacement of production weights.

## Operations

New scripts only, no edits to the installed training package or previous runners.
Shared OS queue/GPU locks with both previous nights, single GPU worker. Save resume
state after every completed detector/classifier epoch. Supervisor deadline stops
only its own worker process tree; do not start new training after deadline. If all
jobs finish earlier, audit/report and disable this night's scheduler immediately.

Reports: experiments_updated_2026_10_02/overnight_2026_10_23/.
Models/logs: outputs_updated_2026_10_02/overnight_2026_10_23/.
Runner: scripts/overnight_architectures.py. Training primitives:
scripts/architecture_components.py. Freeze protocol/configs/input/code hashes
before starting. On confirmed bug, stop for inspection; never silently re-pin
previously executed code/configs. New protocol version needed if scientific
semantics change. Morning report NIGHT_REPORT_2026_10_23.md records actual results,
limitations, weight+threshold paths, and scheduler shutdown.
