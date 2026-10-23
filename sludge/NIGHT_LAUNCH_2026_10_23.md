# Third-night launch and operational handover

Scientific protocol is frozen in `NIGHT_RUN_2026_10_23.md` and
`experiments_updated_2026_10_02/overnight_2026_10_23/protocol.md`.
This file is a technical handover, not a protocol amendment.

## Actual launch

- Started **2026-10-22 18:56:52 Moscow**; deadline **2026-10-23 08:00 Moscow**.
- Runner: `.venv\Scripts\python.exe -u scripts/overnight_architectures.py --run`.
- Initial supervisor wrapper PID32896, interpreter PID19924.
- Initial worker0 wrapper PID52348, interpreter PID50512. These four process
  entries represent ONE supervisor and ONE worker, not four GPU workers.
  Always re-check command lines and creation times; do not trust stale PIDs.
- Logs: `outputs_updated_2026_10_02/overnight_2026_10_23/runner_20261022_185652.stdout.log`
  and the corresponding `.stderr.log`; per-job logs in the same directory.
- At18:58:06: first agnostic detector completed4/30 epochs, epoch4 loss0.3911,
  atomic `fold_0/cascade_s20261022/agnostic/resume_state.pt` present (~345.6MB).
  Training loss is NOT validation quality. No new completed pipeline metrics yet.

## Registration and verification

- Twelve new pipeline jobs,24 component fits; six verified cosine30 controls reused.
- Three folds x two seeds20261022/20261023. Cascade and two-specialist methods;
  union merge is diagnostic, not an additional training job or selectable candidate.
- Registration pins211 inputs (including all204 development image files),18 new
  detector configurations, the six old control configurations and their artifacts,
  new scripts/tests, protocol, old registration and installed-package digest.
- Both shared OS locks were free before launch. The runner acquired queue lock;
  its worker acquires GPU lock. Old scheduled tasks remain disabled.
- Original test is not re-evaluated. Existing metadata/test-usage files are checked
  by hashes, not used to choose this study's recipes or thresholds.
- CPU summary rebuilt from the six old controls matches their existing primary
  mean macroF1 **0.311569** (global_f1 0.301014; fixed_020 0.284463).
  This is adaptive development validation, not independent test performance.

## Tests before launch

- Full regression suite:236 passed,1 deselected (Windows symlink privilege test).
- After final deadline/atomic-output/diagnostic-counter changes:54 relevant tests
  passed, including all19 new architecture tests and existing calibration/queue/
  grouped-split tests. Local COCO-to-ResNet50 initialization and CPU backward pass
  verified; classifier AdamW resume tested exactly on a synthetic model.
- Black check and Ruff E/F/I/B/UP checks passed on the three new Python files.
- Existing installed/source package and old frozen runners were NOT edited.

## Scheduled supervision

New app-owned task, bound to this same thread, every30 minutes:

`Микроорганизмы — каскад и специалисты до 08:00 МСК 23.10`

ID:
`scheduled-task:command:mcp:b09725f3-016d-49ef-9104-56710804ac36:schedule-task:microorganisms-overnight-20261023-v1`

First scheduled check returned by the scheduler: **2026-10-22 19:27:56.835 Moscow**
(`2026-10-22T16:27:56.835Z`). Do not create a duplicate task.

At each check inspect actual time, processes, queue, logs and resume states. Do
not launch a second worker/supervisor. Before deadline, a genuinely interrupted
queue can resume only when both OS locks are free and no job is failed. Use the
same runner, hidden, with new separate stdout/stderr. Failed jobs require explicit
diagnosis; never blind-retry or silently replace registered hashes.

The supervisor stops only its child worker tree at deadline. Workers additionally
check deadline before each detector forward/classifier batch; last completed epoch
resume states survive. The detector retains the frozen30-epoch recipe, and the
classifier the frozen20-epoch recipe, even after resumption.

After all12 jobs finish, or after deadline, run CPU-only
`scripts/overnight_architectures.py --summarize`, verify artifacts, and prepare
`NIGHT_REPORT_2026_10_23.md`. Record all methods/policies, pooled OOF per seed,
mean/SD, class F1 and count MAE/bias, eligibility, localization/duplicates/class
diagnostics, oracle crop confusion, timing/parameter costs, full weight and
calibration paths, incomplete runs and limitations. No automatic deployment.
Disable ONLY this task after the report or user cancellation. Do not reactivate
the21/22October tasks or add further experiments after the fixed budget.
