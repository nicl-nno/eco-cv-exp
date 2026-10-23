# Partial-specialist hybrid extension: operational handover

New user authorization at approximately 00:04 Moscow, 23 October 2026. This is
a separate continuation after the completed cascade/specialists experiment, NOT
a resumption or modification of its frozen registration. No subagents.

## Frozen scientific scope

Read `NIGHT_RUN_2026_10_23_HYBRIDS.md` and
`experiments_updated_2026_10_02/overnight_2026_10_23_hybrids/protocol.md`.
Registration created at **00:15:24.737050 Moscow**. SHA256:
`3c1225c5a81b9dce665885b3fcbd414390f6daf3578284f45c2359671a5d549c`.

- Six fold/seed jobs x five new components = **30 COCO detector fits**, 30epochs
  each. Components: attached, priority, rare, priority_repeat, rare_repeat.
- Twelve fixed class-routing candidates, each evaluated on3folds x2seeds.
- Reuse six cosine30 general controls and12 filamentous/others detectors; no
  retraining or modification. Unchanged classes retain their exact raw predictions
  and PRIMARY count_guarded thresholds. Diagnostic global_f1 uses joint calibration.
- One source/class, no additional global top100 cap or cross-class NMS. Up to400
  outputs/frame for a four-source route; report the output/compute budget change.
- Student_6 primary; original test excluded, never reevaluated. Previously inspected
  outer folds are adaptive development, NOT a new independent test.
- Deadline **2026-10-23T08:00:00+03:00**, today, not24October.
- No additional recipes/subsets/seeds/folds beyond this budget, no deployment.

## Launch and actual processes

Started **2026-10-23 00:19:54.725403 Moscow**, hidden, with separate stdout/stderr:

`.venv\Scripts\python.exe -u scripts/overnight_hybrids.py --run`

Initial supervisor wrapper PID44312, interpreter PID39276. Initial worker0 wrapper
PID52432, interpreter PID53908. These represent ONE supervisor and ONE GPU worker,
not four workers. Inspect live command lines/parent relationships/creation times;
do not trust these PIDs after process termination or reuse.

Shared OS queue.lock/gpu.lock under outputs_updated_2026_10_02/overnight_2026_10_21
were both acquired and released successfully immediately before launch. Existing
supervisor check was empty. Runner/worker acquire their own locks atomically.

Logs:

- `outputs_updated_2026_10_02/overnight_2026_10_23_hybrids/runner_20261023_001954.stdout.log`
- `outputs_updated_2026_10_02/overnight_2026_10_23_hybrids/runner_20261023_001954.stderr.log`
- Per-fold logs: same directory, `fold_N_hybrids_sSEED.log`.
- Queue: same directory, `queue_state.json`.
- Components: `fold_N/hybrids_sSEED/COMPONENT/{model.pt,resume_state.pt,complete.json,train_log.jsonl}`.
- Reports/thresholds: `experiments_updated_2026_10_02/overnight_2026_10_23_hybrids/fold_N/hybrids_sSEED/`.

At00:20:42 the queue was running worker0, remaining five jobs pending. No scientific
metrics for the new candidates yet. First fold has96 usable FIT frames:120 attached,
216 priority-pair and33 rare-group target objects. These are FIT counts, not quality.

At **00:21:46 Moscow**, first attached detector completed **3/30 epochs**; epoch3
loss0.223648 (training loss, NOT validation quality). Atomic `resume_state.pt`
present,345604057bytes, last write00:21:41. Worker log progressing without errors;
GPU utilization61%, memory9165MiB at that instant.

## Preflight verification

- 14 new synthetic tests passed, including end-to-end worker calibration-before-
 outer sequencing, deterministic resumption, strict eligibility, untouched class
 predictions/thresholds, negative FIT frames and no cross-source global cap.
- Full regression suite passed, excluding only the known Windows symlink-privilege
 test `test_symlinked_inputs_keep_project_relative_names`.
- Black and Ruff E/F/I/B/UP checks passed for all three new Python files.
- All old frozen code/package/input/model hashes verified by new registration.
- CPU baseline summary reproduced cosine30 primary macroF1 **0.3115691360871327**,
 SD **0.0036322139031963376**. Current summary initially contains only controls.
- Summary is refreshed by runner AFTER queue completion; while training, new
 pooled metrics need not exist. Do not launch a competing summarize/runner process
 against the active queue. Fold metrics appear when each fold/seed job completes.

## Recurring supervision

New task, this same thread, every **30minutes**:

`Микроорганизмы — частичные специалисты до 08:00 МСК 23.10`

ID: `scheduled-task:command:mcp:bab5f52a-4324-4652-8c6c-3130e8c8c572:schedule-task:microorganisms-overnight-20261023-hybrids-v1`.

First scheduled check: **2026-10-23 00:50:41.764 Moscow**
(`2026-10-22T21:50:41.764Z`). Old21/22/23October tasks remain DISABLED.

Before deadline, interrupted runner may resume only if no active supervisor/worker,
both OS locks free, no failed jobs, with the same frozen runner hidden and new
separate stdout/stderr. Never blind-retry failures or silently re-pin changed code.
Scientific corrections require a new protocol version, not mixed results.

At/after deadline no training; ensure only our worker tree stops and last full
epoch checkpoints survive. Supervisor may continue permitted CPU summary work.
Never kill unrelated processes. Before any restart check actual date/time.

After completion/deadline and no active queue, run:

```powershell
$env:PYTHONUTF8='1'
.venv\Scripts\python.exe -u scripts/overnight_hybrids.py --summarize
.venv\Scripts\python.exe -u scripts/report_hybrid_night.py --reuse-summary
```

The report utility reads resume checkpoints on CPU, audits all completed/partial
components, verifies untouched classes and writes final_integrity.json,
weight_index.json and final_metrics.md. If summary is missing/stale, omit
`--reuse-summary`; it will rebuild it under the shared OS locks.

Write `NIGHT_REPORT_2026_10_23_HYBRIDS.md`: all12 candidates and all policies, pooled
OOF separately per seed then mean/sampleSD, class F1/P/R, count MAE/bias with/without
ignore, strict eligibility (unchanged), localization/duplicates, costs, weights and
thresholds, partial runs, limitations and next steps. Include old/reused timing
caveats and multiple adaptive comparisons. No release superiority claim or deploy.

Then find the exact new title via list_scheduled_tasks and disable ONLY this task
with update_scheduled_task(enabled:false), verifying nextRunAt=null. Also disable
on user cancellation. Finish early if the registered budget completes early.
