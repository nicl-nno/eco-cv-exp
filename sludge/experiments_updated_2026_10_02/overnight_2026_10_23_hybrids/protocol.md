# New authorized extension: selected-class specialists, v1

User authorized a new bounded overnight study at approximately 00:04 Moscow on
23 October 2026, after completion of the previous architecture study. This is a
NEW registration, not a change to any old experiment. Deadline: **2026-10-23
08:00:00+03:00**. No subagents, test evaluation, deployment, paid services,
publication, commits or PRs. Student_6 remains primary (`prefer_student_6`).

## Hypothesis and fixed data

Replace only selected classes of the general cosine30 detector with specialist
outputs. Preserve predictions AND PRIMARY thresholds of untouched classes exactly.
Do not reject overlaps between different classes based on incomparable raw scores.
This tests selective specialization, not the failed localization/classifier cascade.

Reuse the six frozen cosine30 controls, six filamentous and six others detectors
from the completed study. Verify hashes; do not fit them again. Reuse the exact
three FIT/calibration/outer group partitions and seeds 20261022/20261023: 204 outer
development frames, 31 groups. Original test IDs/groups/hashes stay excluded.
These folds have already been inspected: this is adaptive development with
multiple comparisons, NOT independent testing or proof of release superiority.

## Fixed new training budget

Six fold/seed jobs, each with FIVE new component fits = **30 detector fits**.
Every detector starts from local COCO weights, final epoch30, same cosine30
resize/augmentation/LR/frozen-BN/exclude-ignore recipe. No new downloads.

1. `attached`: attached_ciliates only.
2. `priority`: attached_ciliates + filamentous_bacteria.
3. `rare`: free_swimming_ciliates + nematoda + gastrotrichs.
4. `priority_repeat`: same labels as priority, registered mild FIT-only
   `repeat_sqrt_020_cap3` sampler, N replacement draws/epoch.
5. `rare_repeat`: same labels as rare, same repeat sampler.

All usable FIT images remain, including images with no selected targets. Remap
only in memory; never rewrite prepared annotations. Log FIT label support and
sampling weight ranges (weights can all be1; replacement still differs from
shuffle). Only sampling differs between the paired recipes. Last full epoch
checkpoint after each epoch; no outer-based model/epoch selection.

## Fixed candidate routes

Every class has exactly ONE source. All classes not explicitly listed use the
general control. The old others detector is used only for the common-three route.

- `h_filamentous`: filamentous from old binary specialist.
- `h_attached`: attached from new attached.
- `h_priority`: attached/filamentous from new priority.
- `h_separate`: attached from new attached, filamentous from old binary.
- `h_rare`: rare-three from new rare.
- `h_priority_rare`: priority pair plus rare-three.
- `h_separate_rare`: separate priority specialists plus rare-three.
- `h_common3`: rotifers/testate_amoebae/gastrotrichs from old others.
- `h_priority_repeat`: priority pair from repeat variant.
- `h_rare_repeat`: rare-three from repeat variant.
- `h_priority_repeat_rare_repeat`: priority repeat + rare repeat.
- `h_separate_rare_repeat`: separate priority specialists + rare repeat.

Twelve candidate definitions, six fold/seed realizations each. No search over
additional class subsets after results. Within-source score floor.05, NMS.5,
max100 predictions unchanged. **No additional global top100 cap or cross-class
NMS**: these would change untouched classes and compare unlike confidence scales.
Output may exceed100/frame (at most400 for a four-source route); report this
budget difference and cross-class duplicate/true-overlap limitations. There is no
union of multiple sources for the same class and no post-hoc source selection.

## Calibration and outcome rules

Compute source-specific calibration using the unchanged old calibrate function
on that source's named classes, filtering other target annotations but retaining
ignore regions and every calibration image. The support fallback operates on
the source's own global threshold. General control selection is reused unchanged.
PRIMARY `count_guarded` routes each class's threshold from its SAME source.
This source-preserving PRIMARY differs from recalibrating a new global fallback
for the entire hybrid; that distinction is intentional and declared in advance.
Diagnostics: fixed_020 and a common global_f1 recalibrated on the complete hybrid.
All thresholds and routes are frozen before new outer inference or loading reused
outer predictions for hybrid evaluation. Historical outer inference obviously
predates this new study; it is not presented as a new blind evaluation.

Evaluate all candidates, no post-hoc winner-specific tuning. Pooled OOF separately
per seed, then mean/sample SD over two seeds, never concatenate duplicated frames
across seeds. Report class F1/precision/recall, count MAE/bias with/without ignore,
support, localization/duplicates, time and parameter budgets. Raw localization
diagnostics at score.05 are not matched-operating-point comparisons.

Keep the existing STRICT eligibility: mean macroF1 >= control-.01; both attached
AND filamentous MAE <=90% control; no class F1 drop >.02. Untouched priority classes
therefore cannot meet the10% gain condition; report failure honestly. Also report
per-class deltas to understand partial successes, but do not redefine eligibility.
No deployment even if eligible; reused adaptive folds require fresh verification.

## Operations and reproducibility

New runner `scripts/overnight_hybrids.py`; old code, package, configs, registrations,
models and summaries remain unchanged. Freeze this protocol, new source/tests,
input/model/prediction/config hashes and package digest before fitting. Shared
OS locks: outputs_updated_2026_10_02/overnight_2026_10_21/queue.lock and gpu.lock.
One supervisor, one GPU worker. Windows wrapper+child is one worker. No blind
retry of failed jobs or bypass of checksum guards. Resume only the same frozen
queue before deadline when both locks are free and no failed jobs exist.

Supervisor terminates only its child tree at deadline; detector forward hooks
also enforce cutoff. Last full epoch resume states survive. After budget/deadline
no new jobs. CPU summary allowed afterward. Report complete and partial results,
write NIGHT_REPORT_2026_10_23_HYBRIDS.md, verify checkpoints/processes, and disable
only the new hybrid recurring task. Old tasks stay disabled.

Limitations: annotation incompleteness and annotation-relative negatives, rare
class support, only two seeds, larger compute, scores not cross-source calibrated,
real overlapping organisms and duplicate physical objects across class sources.
Class groups and hypotheses are informed by prior outer results. No independent
test claim, no claim of superiority to release (release train IDs unknown).
