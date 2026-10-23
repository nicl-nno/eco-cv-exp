# SAM-family FIT-only feasibility pilot v1 — 23.10.2026

New user-authorized hypothesis, not an extension of the closed overnight budget.
Inference only; no fine-tuning, no optimizer, no deployment. No subagents.

Models: official SAM ViT-B (Apache-2.0, segment-anything commit
dca509fe793f601edb92606367a655c15ac00fdf) and EfficientSAM ViT-T (Apache-2.0,
commit d525f622e6f640acf5a0fc37c7ca1f243da5bde0). Only published pretrained
weights; record weight SHA256 and source pins before inference. EfficientSAM
weight also checked against its Git blob object at the pinned commit.
SAM2 is a later candidate, not part of this two-model pilot. No environment
package install/change, no cloud service, no upload of images.

## Data / budget

Exactly 12 unique usable FIT frames from fold0/cosine30/seed20261022 config,
Student_6 priority. Select using annotations/IDs only, before any model outputs:
sort frames by SHA256('sam-pilot-v1:' + image_id); visit classes from rarest FIT
image support to commonest, choosing the first not-yet-selected group containing
that class; then fill to12 with unused groups in hash order. One image per group.
No performance-based sample replacement. Missing rare-class coverage is disclosed.
No ignore frames. Verify FIT/calibration/outer separation and exclude original
test IDs/groups/file/pixel hashes via metadata; never read test pixels.
Register image file hashes and GT boxes/labels. No selection from calibration or
outer, and no comparison to existing pooled OOF numbers as if same sample.

2 models ×12 frames =24 image embeddings; a maximum20minutes inference wall
budget checked between batches. One GPU worker under existing queue.lock and
gpu.lock. Do not touch unrelated GPU processes. No training or scheduled task.

## Fixed inference modes

Use original RGB frames and upstream model preprocessing. FP32, eval,
inference_mode, fixed seed20261023. Cache one embedding per model/frame.
Prompt batches of8. No image crop pyramid or iterative mask refinement.

1. Automatic: 16×16 uniform positive point grid at cell centers, no GT prompts.
   Decode3 candidates per point. Mask threshold logit0. Pixel boxes use exclusive
   xmax/ymax. Minimum area16px, box NMS IoU0.7, top100 by predicted mask IoU.
   PRIMARY feasibility operating point: predicted IoU>0.88, stability>=0.95
   (intersection at logit+1 / union at logit−1). These are fixed upfront,
   NOT calibrated thresholds. DIAGNOSTIC raw_top100 omits quality/stability
   cutoffs but uses the same area/NMS/cap. No threshold search after results.
2. GT-center prompt for each annotated object, choose candidate with largest
   model-predicted IoU. No GT-overlap-based candidate selection. Bbox center
   need not lie inside a filament; disclose this limitation.
3. GT-box prompt for each annotated object, same candidate selection rule.
   This is an oracle-prompt diagnostic, NOT automatic localization/counting.

Identical automatic filtering across models is a bounded common operating point,
not proof that their quality scores are calibrated alike. Record oracle modes
separately from automatic. No inherited class prediction: SAM masks are class
agnostic. No invented microorganism classifier or post-hoc routing.

## Metrics and interpretation

Automatic: class-agnostic box matching at IoU0.5, greedy descending model score,
one prediction/GT match. Report precision/recall, TP/FP/FN, duplicate FP, proposal
coverage, per-GT-class localization recall, total count MAE and signed bias per
frame. Count all retained proposals, including debris; annotations are incomplete.
GT categories in recall do NOT imply predicted categories. No class-F1 claim.

Oracle: returned mask-bounding-box IoU with its supplied object's GT box, mean
and fraction>=0.5; report separately point and box prompts. **No mask ground truth
is available in prepared annotations: cannot report mask IoU/Dice, boundary quality
or true instance-segmentation accuracy.** Oracle box consistency is particularly
optimistic because the model receives GT geometry. Oracle count is not evaluated.

Save every frame's boxes/scores/areas, timing, parameter count and peak allocated
GPU memory. Save annotated preview images only for the first2 registered frames,
for qualitative QA; do not change choices based on previews. Existing FIT data
have already been used for development; results are feasibility diagnostics,
not independent test or validation and not a reason to deploy.

No eligibility decision against cosine30: this pilot has neither class predictions
nor comparable OOF coverage. Preserve original strict eligibility for any future
registered end-to-end comparison. Follow-up (not in this budget): detector-box
prompts, cell-specific segmentation models or SAM2, with separate protocol.

Freeze protocol, runner/tests, model/source/input pins and selection in
registration.json before inference. Failed/started runs cannot be blindly rerun;
diagnose first, preserve partial outputs. Scientific changes require a new version.
