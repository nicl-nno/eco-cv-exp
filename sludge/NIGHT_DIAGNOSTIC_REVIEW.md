# Validation diagnostic review pack — 21 October 2026

Research model: `cosine30_s20260929` (first registered seed, not best-seed selection).
Operating confidence: its previously selected **0.60**, not the CV protocol's 0.20.
This pack reuses saved validation predictions; it does not evaluate test or run inference.

## Files and interpretation

- Index: `experiments_updated_2026_10_02/overnight_2026_10_21/cosine30_s20260929/examples.json`.
- Images: `outputs_updated_2026_10_02/overnight_2026_10_21/cosine30_s20260929/examples/`.
- 21 selected objects on 17 frames: seven correct matches (green), seven false positives
  against current annotations (red), seven missed reference boxes (blue).
- The images show only selected example boxes, NOT all predictions or annotations.
  Consult the JSON for the outcome of each box; the image label itself omits outcome text.
- Correct and missed examples alternate between available classes; false positives are
  selected by highest confidence. This selection is deliberately diagnostic, not a
  random or representative sample. Absence of a class from correct examples is not a
  separate numerical estimate of recall.

Two images (`student_06-0018.jpg`, `student_07-0026.jpg`) were inspected for rendering.
This checks readability, not organism identity or annotation correctness.

## Questions for a microscopy expert

For each red box, distinguish a genuine wrong class/background detection from an
unannotated organism, inaccurate reference box, or multiple detections of one organism.
For each blue box, check reference class and extent, visibility/occlusion, and whether
the image supports identification at this taxonomic resolution. Preserve the original
labels and record proposed corrections separately with reviewer and reason. Student_6
precedence for conflicting duplicate source annotations remains unchanged.

No relabeling or threshold tuning has been done from this pack. Any subsequent expert
adjudication must version the dataset and protocol; it cannot silently replace the
labels underlying the reported overnight results. Before deployment, prioritize rare
class recall and the observed attached-ciliate/filamentous-bacteria count overestimates,
then validate on newly collected independent groups. This pack alone cannot establish
which biological or annotation failure mechanism causes those aggregate errors.
