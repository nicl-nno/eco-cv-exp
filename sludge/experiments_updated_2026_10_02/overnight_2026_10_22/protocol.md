# Second night: calibrated counting and rare-class development study

User authorized another overnight improvement series at 21:31 Moscow, 21 October.
Deadline: **2026-10-22T08:00:00+03:00**. No subagents, deployment, commits, PRs,
paid services or data publication. This protocol is recorded BEFORE new training.
Do not modify first-night configs/results. Never evaluate the original test again.

## Archive check and frozen data

Root `../7.zip` and `../10.zip` pass ZIP CRC and exactly match the already present
UPDATED data archives, not the missing historical originals:

- 7: `d0b307659288869c6c31d93227a7b6cc629ccde47c5df7f6d5ea2d5d610a10c9`.
- 10: `11032a172ba4bc8bcac69220ffc34733c182c6169d5a92c40d2c5643c330dcad`.

Leave root files intact. Reuse the byte-identical copies in updated_2026_10_02;
do not substitute them for original Student_7/Student_10. Historical release split
reconstruction therefore remains blocked and is NOT a prerequisite for this study.
The full updated dataset and Student_6 conflict precedence remain unchanged.
Original split SHA: `6f88c2d3404b9028ea0fa38b81ac4ebaab19d6881d83a9c426e3e2c2ff09675c`.

## Registered questions and budget

Exactly **24 training jobs**: four recipes x three existing outer development folds
x seeds 20261022, 20261023. All initialize from the same local COCO checkpoint.

1. control15: original 15 epochs, no augmentation, constant LR.
2. cosine30: first night's rotations/flips, cosine LR, 2 warmup epochs, 30 epochs.
3. repeat30: cosine30 plus capped square-root repeat-frequency sampling.
4. repeat60: repeat30 for 60 epochs, schedule stretched to 60 epochs.

Repeat weights use only usable FIT samples. For each class with image frequency f,
weight is max(1, sqrt(0.20/f)); image weight is the maximum over its target classes,
capped at 3. Empty/ignore-only images receive weight 1. Draw exactly N images per
epoch with replacement. This is a milder alternative to prior inverse-frequency
sampling; a hypothesis, not an asserted improvement. All other training/model
settings stay fixed, frozen BN, base LR 0.0025, final LR factor 0.1. Final epoch
checkpoints only; no adaptive stopping or cherry-picked best seed.

## FIT / calibration / outer validation isolation

Reuse the registered outer folds from grouped_cv_2026_10_21 (204 original train+val
frames, 31 groups). Original test groups/image IDs/pixel and file hashes excluded.
For each outer fold, partition ONLY its training groups into FIT and calibration,
target 80/20. Use 500 deterministic candidate group subsets, seed 20261022+fold,
balancing frame and class counts; reject any split that removes a target class
from usable FIT when that class exists in outer training. No model predictions
are used to assign groups. Freeze manifests, hashes, group IDs and class support
BEFORE training. Preserve original group identity and dHash separation.

The model trains on FIT only (ignore-containing FIT frames excluded as before).
Its ordinary validation metrics refer to calibration only, NOT outer validation.
Calibrate thresholds from saved calibration predictions, freeze them to JSON,
then run inference on the outer validation frames. Never refit on calibration
before evaluating the frozen thresholds; no exposure to outer labels in threshold
selection. Guard every new experiments folder against any test evaluation.

## Fixed evaluation policies, all reported (no operating-point cherry picking)

IoU 0.50, existing ignore policy and postprocessing unchanged. Each model produces:

- fixed_020: confidence 0.20 for all classes, diagnostic bridge to the first night.
- global_f1: threshold maximising calibration macro F1 on 0.05:0.05:0.95;
  ties go to higher threshold. Fallback 0.20 if calibration has no defined F1.
- count_guarded (PRIMARY): per-class calibration on the same grid. Only specialize
  classes with at least five calibration objects in at least three calibration
  groups and nonzero best F1. Among thresholds with F1 >= 90% of that class's best
  calibration F1, minimize count MAE including all calibration frames; break ties
  by higher F1, then higher threshold. Other classes use global_f1. Save support,
  group support, fallback reason and selected thresholds before outer inference.

Report seven-class F1/precision/recall, counts MAE/bias both with and without ignore
filtering, per-fold results and pooled OOF per seed. Compute across-seed mean/SD
from the two separately pooled OOF sets, not duplicated image IDs. Keep all failed
and partial jobs in the ledger; no blind failure retry.

An improvement is eligible for follow-up only if, on PRIMARY mean pooled results,
macro F1 is >= cosine30 minus 0.01, BOTH attached_ciliates and filamentous_bacteria
count MAE improve by >=10%, and no class F1 regresses by >0.02 vs cosine30.
Also report control15 and all policies even if the criterion fails. No deployment
decision or superiority-to-release claim from this study alone. Stop after the
registered budget; no new recipes/threshold grids/folds in response to results.

## Interpretation and supervision

These outer folds were already inspected in the first night and influenced the
new hypotheses. Inner calibration avoids direct threshold tuning on outer frames,
but does NOT turn this adaptive development study into a new independent or fully
nested generalization test. Rare-class calibration support may be insufficient.
The amount of FIT data is lower than the first night, so numerical comparisons
across nights are descriptive; within-night controls use identical FIT sets.

Only one GPU worker at a time, sharing original OS queue/GPU locks. Atomic resume
state after each epoch; deadline supervisor terminates only its own worker tree.
After 08:00 do not train. Verify saved weights, report actual metrics/limits and
disable this night's recurring task. CUDA bitwise determinism is not promised.

Runner: `scripts/overnight_calibrated.py` (prepare, run, worker modes).
Reports: `experiments_updated_2026_10_02/overnight_2026_10_22/`.
Runtime/logs/models: `outputs_updated_2026_10_02/overnight_2026_10_22/`.
