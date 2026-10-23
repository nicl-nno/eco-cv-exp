# Registered follow-up: grouped development cross-validation

Registered while the first overnight queue is still running (21 October 2026).
Do not start this stage until all 12 first-stage jobs have finished successfully.
Do not modify the first-stage jobs, training package, or existing split.

## Selection and budget

- Select the non-control recipe with the highest mean validation macro F1 over
  all three registered seeds. Break exact ties by the recipe name alphabetically.
- Proceed only if its mean is greater than the control15 mean; otherwise report
  no confirmed development gain and do not spend this stage's training budget.
- Compare that recipe against control15 on three grouped development folds,
  with seed 20261021 for both methods in each fold: at most six training jobs.
- Use the recipe's original final-epoch duration and COCO initialization, no
  additional hyperparameter tuning. Exclude ignore-containing training frames.
- Keep the overall night deadline 08:00 Moscow, 21 October 2026. One GPU worker
  at a time, using the same shared GPU lock as the overnight queue.

## Membership and independence

Use ONLY the original train+validation frames (204 frames, 31 scene groups).
Exclude every original test group, image ID and pixel hash. Scene groups are
inherited from the original locked manifest, not reconstructed using predictions.
Balance three folds with the existing deterministic group allocation algorithm,
equal fractions, seed 20261021, 200 restarts. Features include frame count,
class object/image counts, and class counts after excluding ignore-bearing frames.
The three allocation labels used internally by the allocator are just fold indices;
none is the original test. Each resulting training manifest contains only train
and validation rows and no test rows whatsoever. Check group/hash/dHash separation.

Freeze the memberships and hashes before any follow-up training. Each development
frame serves as validation once; its whole scene group stays outside that fold's
training. Inspect per-fold training support and explicitly disclose any missing class.

This is a development robustness check, NOT an untouched external test and NOT
nested cross-validation: the candidate was selected using the original validation,
whose frames now participate in these folds. Do not claim unbiased generalization
or independent superiority over released weights from this stage.

## Metrics fixed before training

Primary comparison: pooled out-of-fold macro F1 at confidence **0.20**, fixed before
training, identical for both methods (the release's documented operating point).
Also report per-fold F1, seven class F1/support values, and count MAE/bias per image.
Use existing IoU 0.50 and ignore-region rules. Include all validation frames in count
metrics, including zero-detection frames. Training's automatic validation-tuned
threshold reports are diagnostic only and MUST NOT replace this fixed-threshold
comparison or be described as unbiased fold performance.

No test evaluation, passport replacement, deployment, or release-weight fine-tuning.
Keep all six checkpoints and raw validation predictions. Stop after this comparison
and report uncertainty; no unregistered search over folds, thresholds or seeds.

## Execution

The runner is `scripts/run_grouped_cv.py`. It refuses incomplete first-stage results,
checks each first-stage checkpoint/configuration manifest, applies the selection rule,
then freezes `selection.json` and the six job YAML files. It writes class-level and
counting comparisons to `first_stage_analysis.json`. Preparation is CPU-only:

```powershell
$env:PYTHONUTF8='1'
.venv/Scripts/python.exe scripts/run_grouped_cv.py --prepare
.venv/Scripts/python.exe -u scripts/run_grouped_cv.py --run
```

Launch `--run` in a hidden background process with separate stdout/stderr logs only
after confirming the original queue has exited. It also takes the ORIGINAL queue
lock and shared GPU lock, so an overlapping launch is refused even between workers.
The six jobs run sequentially with atomic state and per-epoch recovery checkpoints.
The old and new queues must never both hold active GPU workers.

Reports: `experiments_updated_2026_10_02/grouped_cv_2026_10_21/`.
Runtime state/logs/checkpoints: `outputs_updated_2026_10_02/grouped_cv_2026_10_21/`.
`summary.json` in reports contains fixed-threshold per-fold metrics and, only after
all six jobs finish, pooled out-of-fold metrics and the candidate-minus-control delta.
There is no CV model deployment step. Inspect failures before retrying, keep prior logs,
and never change frozen job configurations or fold manifests.
