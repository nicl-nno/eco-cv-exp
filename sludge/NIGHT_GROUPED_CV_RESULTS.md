# Grouped development CV results, 21 October 2026

All six pre-registered jobs completed successfully. This comparison uses 204
development frames / 707 annotated target objects from 31 scene groups. Each frame
is predicted once by a model that did not train on its scene group. Original test
frames/groups are excluded; no test evaluation was performed.

Candidate: cosine30 (selected from the prior three-seed development study).
Control: original 15-epoch training recipe, retrained from COCO per fold.
Confidence **0.20 was fixed before these runs**, identical for both methods;
IoU matching 0.50 and existing ignore-region rules are unchanged.

## Detection

| Fold | Validation frames | Control macro F1 | Cosine30 macro F1 |
| --- | ---: | ---: | ---: |
| 0 | 67 | 0.21986 | 0.31774 |
| 1 | 75 | 0.24499 | 0.32014 |
| 2 | 62 | 0.27522 | 0.31508 |
| Pooled out-of-fold predictions | 204 | **0.24608** | **0.31620** |

The pooled difference is **+0.07013 absolute macro F1**. Pooling combines TP/FP/FN
counts per class before computing F1; it is not the arithmetic average of fold F1.
The candidate wins the primary detection metric in all three folds.

## Detection is not the same as reliable counting

Count MAE below is objects per frame, including frames with no objects/detections.
Detections within annotated ignore regions are excluded here; raw-output count errors
are also preserved in the machine-readable report.

| Class | Objects | Control F1 | Candidate F1 | Control count MAE | Candidate count MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| attached_ciliates | 231 | 0.242 | 0.253 | 2.059 | **3.275** |
| filamentous_bacteria | 235 | 0.231 | 0.259 | 1.975 | **3.240** |
| rotifers | 78 | 0.586 | 0.667 | 0.363 | 0.299 |
| testate_amoebae | 85 | 0.429 | 0.577 | 0.574 | 0.387 |
| free_swimming_ciliates | 28 | 0.034 | 0.112 | 0.505 | 0.358 |
| nematoda | 23 | **0.000** | **0.000** | 0.196 | 0.250 |
| gastrotrichs | 27 | 0.200 | 0.346 | 0.216 | 0.221 |

At this fixed threshold the candidate counts 861 attached ciliates for 231 references
and 876 filamentous bacteria for 235 references. Mean biases are +3.088 and +3.142
objects/frame respectively. Nematode detections match none of the 23 references.
Higher recall/F1 therefore does not establish a usable seven-class counting system.
The deployment decision is **do not replace the delivered model automatically**.

The original validation experiment used per-run optimized thresholds (0.60–0.75 for
cosine30), whereas this registered CV uses 0.20. Its lower threshold and different
frames help explain why counting behavior differs; do not mix the two protocols or
retroactively replace the CV operating point with an optimized threshold.

## Limits and next scientific steps

- This is a development robustness check, not nested CV or a new independent test:
  original validation data influenced candidate selection before these folds.
- One training seed per CV fold, small and imbalanced classes. Fold 1 contains one
  validation nematode. Training sets overlap across folds; folds are not independent
  repeated datasets. No generalization confidence interval is claimed.
- Annotations may be incomplete. False positives and count errors are measured
  against current annotations, not expert-adjudicated biological ground truth.
- The original release was not evaluated on these CV pools because its training
  overlap is unknown. Its 0.388943 on the original current validation is not a
  comparable CV number. Superiority over released weights is still unproven.
- Next improvement should jointly address count calibration and rare-class recall,
  with thresholds calibrated on separate group-held-out data and a new independently
  annotated evaluation set. A higher F1 alone is not an acceptance criterion for
  the aggregate diagnostic table. More diverse nematode/free-swimmer labels are needed.

No extra tuning is performed on these folds. The registered training budgets (12
first-stage jobs + 6 CV jobs) are exhausted. Remaining overnight work focuses on
provenance, verification and a clear handover, not adaptive search against these results.

Machine-readable report:
`experiments_updated_2026_10_02/grouped_cv_2026_10_21/summary.json`.
Fold checkpoints remain under
`outputs_updated_2026_10_02/grouped_cv_2026_10_21/fold_<0|1|2>/<control15|cosine30>/model.pt`.
They are research fold models, not a newly validated deployment model.
