# First overnight stage: development results, 21 October 2026

All 12 pre-registered jobs completed successfully by 02:23:37 Moscow.
Each uses the same 132 training frames, 50 validation frames / 184 reference objects,
Student_6-preferred annotations, architecture, COCO initialization, and evaluation.
Each run chooses confidence on this validation. These are development metrics,
not independent test results. Standard deviations below are across three training
seeds, NOT confidence intervals for generalization to new microscopy samples.

| Training recipe | Mean macro F1 | Seed SD |
| --- | ---: | ---: |
| Original recipe, 15 epochs, no augmentation | 0.29210 | 0.02014 |
| Rotations/flips, 15 epochs | 0.26486 | 0.01852 |
| Rotations/flips, 30 epochs, constant LR | 0.30470 | 0.00680 |
| Rotations/flips, 30 epochs, cosine LR + warmup | **0.33968** | 0.01458 |

Cosine30 improves over the retrained control by 0.04758 absolute macro F1 on average.
Paired seed differences: +0.03184, +0.02522, +0.08568. The pre-registered selection
rule therefore selects cosine30 for the grouped development follow-up.
This does not isolate the effect of LR scheduling alone: compared with control,
augmentation and training duration also differ.

## Class behavior and counting tradeoffs

Metrics below average the three seeds at their individual validation-selected thresholds.
Count MAE is objects per image, including zero-detection images; detections inside
annotated ignore regions are excluded here. Reports also retain unfiltered count MAE.

| Class | Validation objects | Control F1 | Cosine30 F1 | Control count MAE | Cosine30 count MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| attached_ciliates | 67 | 0.211 | 0.259 | 1.907 | 1.713 |
| filamentous_bacteria | 60 | 0.243 | 0.263 | 2.200 | 1.567 |
| rotifers | 20 | 0.591 | 0.687 | 0.260 | 0.213 |
| testate_amoebae | 21 | 0.521 | 0.684 | 0.493 | 0.267 |
| free_swimming_ciliates | 7 | 0.108 | **0.000** | 0.193 | 0.140 |
| nematoda | 2 | 0.000 | **0.000** | 0.080 | 0.040 |
| gastrotrichs | 7 | 0.372 | 0.485 | 0.167 | 0.100 |

The apparent count-MAE improvement for rare classes does NOT imply useful detection:
predicting little or nothing can reduce MAE when most frames contain none. Free-swimming
ciliates regress to zero F1 across all three cosine30 runs; nematodes remain undetected.
The candidate also still overcounts attached ciliates (+1.033 objects/image mean bias)
and filamentous bacteria (+0.553 objects/image mean bias). It is not ready for reliable
seven-class quantitative diagnostics or automatic replacement of the delivered model.

## Existing release comparison

The actual release checkpoint gives macro F1 **0.388943** on the same current validation
at its original confidence 0.20 (also the best threshold on this validation grid).
The new candidate's mean 0.339679, and even best seed 0.354797, do not exceed it.
However, the release's original training membership is unknown: current validation may
overlap its training. This is descriptive, not independent evidence of generalization.
Historical release test/validation metrics use different annotations/splits and must
not be mixed into this same-data comparison.

## Next registered check and artifacts

The three-fold development robustness comparison starts at 02:39 Moscow, after the
first queue exited. Six jobs: control15 and cosine30 on each fold, seed 20261021.
Primary metric uses confidence 0.20 fixed in advance, not fold-optimized thresholds.
All 46 original test images and all their scene groups are excluded. Candidate selection
already used original validation, so this is not nested CV or an untouched test.
See [GROUPED_CV_2026_10_21.md](GROUPED_CV_2026_10_21.md).

Machine-readable sources:

- `experiments_updated_2026_10_02/overnight_2026_10_21/summary.json`
- `experiments_updated_2026_10_02/overnight_2026_10_21/release_baseline.json`
- `experiments_updated_2026_10_02/grouped_cv_2026_10_21/first_stage_analysis.json`
- `experiments_updated_2026_10_02/grouped_cv_2026_10_21/selection.json`

Candidate weights for all seeds are retained under
`outputs_updated_2026_10_02/overnight_2026_10_21/cosine30_s<seed>/model.pt`.
No release weights, delivered passport, deployment, or test results were overwritten.
