# Final Report: Sheet Music Reader

## Project Summary

Sheet Music Reader is a computer vision system that takes a printed sheet music image
as input and produces playable audio as output. The core CV problem is music object
detection where the model works on localizing and classifying 30 categories of musical symbols (noteheads,
clefs, rests, time signatures, key signatures, beams, flags, accidentals) on full
sheet music page images using bounding box detection.

The pipeline has five stages:
1. Detection — YOLOv8s or RT-DETR-L detects all musical symbols on the page
2. Staff detection — horizontal projection profiles locate all staff lines
3. Music theory post-processing — detections are converted to pitch, rhythm, and key information using clef-specific lookup tables and key signature inference
4. MIDI assembly — a two-part score (treble + bass) is assembled using music21
5. Audio synthesis — MIDI is rendered to WAV using FluidSynth

Live demo: [Hugging Face Spaces](https://huggingface.co/spaces/samyu-vakkalanka/sheet-music-reader)

---

## Dataset

*eepScores V2 (dense subset)
- 1,714 images of rendered sheet music (400 DPI, synthetically generated via Lilypond)
- 889,833 annotated symbol instances across 115 classes (filtered to 30 in-scope classes)
- Train/val split: 1,362 / 352 images
- License: CC BY 4.0
- [Zenodo link](https://zenodo.org/record/4012193)

---

## Models and Results

Three models were trained and evaluated on the DeepScores V2 dense val set (352 images, 220,356 annotated instances):

| Model | Params | imgsz | mAP@0.5 | mAP@0.5:0.95 | stem | ledgerLine | augDot |
|---|---|---|---|---|---|---|---|
| HOG + SVM (classical) | — | crop | 94.78% acc | — | F1=0.77 | F1=0.92 | F1=1.00 |
| YOLOv8s (Check-in 2) | 11.1M | 640px | 0.489 | 0.275 | 0.000 | 0.000 | 0.000 |
| **YOLOv8s (ablation)** | **11.1M** | **1280px** | **0.594** | **0.445** | **0.000** | **0.000** | **0.134** |
| RT-DETR-L (advanced) | 32.9M | 640px | 0.281 | 0.189 | 0.000 | 0.000 | 0.004 |

### Key Findings

- Resolution beats architecture: Doubling inference resolution from 640px to 1280px yielded a 21% relative improvement in mAP@0.5 with the identical YOLOv8s architecture. This was the single most impactful intervention across all experiments.

- Transformer underperformed on this dataset: RT-DETR-L (32.9M parameters, AIFI transformer encoder) underperformed YOLOv8s on every class. The most likely cause is dataset size. 1,362 training images is insufficient for a 32.9M parameter model to learn meaningful attention patterns. The full DeepScores V2 dataset (255k images) would likely close this gap.

- Stem and ledgerLine detection remains unsolved: All three deep learning models achieved exactly zero mAP@0.5 on stem and ledgerLine. This is an annotation-level problem. Stem bounding boxes in DeepScores V2 are 1-2px wide, making it impossible for any box-based detector to achieve sufficient IoU overlap. This is consistent with results reported in the original DeepScores V2 paper.

- CNN feature pyramids suit dense local detection: Sheet music detection is a fundamentally local-feature problem. Symbol identity is determined by shape and position, not global context. YOLOv8's feature pyramid network is well-matched to this structure.

---

## End-to-End Pipeline

The MIDI conversion pipeline (`src/midi_converter.py`) converts YOLOv8 detections to
audio through the following steps:

1. Staff lines detected via horizontal projection profiles
2. Each notehead assigned to nearest staff by y-coordinate
3. Clef determined by most confident clefG/clefF detection per staff
4. Key inferred by counting keySharp/keyFlat detections → lookup table
5. Time signature parsed from timeSig detections (defaults to 4/4)
6. Pitch assigned via staff position → clef-specific lookup table
7. Rhythm assigned via notehead type + nearby beam/flag detections
8. Two-part score (treble + bass) assembled via music21 and exported to MIDI
9. MIDI rendered to WAV via FluidSynth

**Known limitations of the MIDI pipeline:**
- Time signature defaults to 4/4 because 3/4 detection is unreliable (timeSig3 mAP = 0.112)
- Stem detection failure means rhythm inference relies on notehead type and beams only
- Pitch calibration tuned for standard engraving sizes and may drift on unusual layouts
- Ties, slurs, dynamics, ornaments, and repeat signs not handled

---

## Failure Modes

### Detection failures
- Stem / ledgerLine (all models, mAP = 0.000) — annotation-level limitation, bounding boxes too thin for box-based detection
- Beam regression at 1280px (0.595 → 0.363) — wide thin boxes penalized more heavily by IoU threshold at higher resolution; localization not classification issue
- Rare classes — timeSig2 (274 instances), timeSig3 (823 instances) remain weak due to data scarcity regardless of model or resolution

### Pipeline failures
- 3/4 time not reliably detected — most common rhythm error in audio output
- Duplicate notehead detections — overlapping bounding boxes cause some notes to play twice
- Real-world generalization — dataset is fully synthetic; real scanned sheet music with noise, skew, or degradation may reduce detection quality

---

## Limitations and Next Steps

Highest priority:
- Train at full resolution on the complete DeepScores V2 dataset (255k images) which would address data scarcity for rare classes and potentially enable RT-DETR to compete
- Implement tiled inference for full-resolution detection without downscaling — would address stem/ledgerLine failure
- Add non-maximum suppression on noteheads to eliminate duplicate detections

**Medium priority:**
- Add brace detection to enable proper multi-staff system grouping
- Improve time signature detection reliability
- Test on real scanned sheet music and add augmentation to bridge the domain gap

Longer term:
- Skeleton-based or keypoint-based annotation for stems and ledger lines
- Fine-tune on a beginner/intermediate piano score subset aligned with target use case
- Extend MIDI pipeline to handle ties, repeats, and dynamics

---

## Repository Navigation

| File | Location |
|---|---|
| EDA notebook | [notebooks/eda.ipynb](../notebooks/eda.ipynb) |
| Classical baseline | [notebooks/classical_baseline.ipynb](../notebooks/classical_baseline.ipynb) |
| CNN baseline (YOLOv8s 640px) | [notebooks/cnn_baseline.ipynb](../notebooks/cnn_baseline.ipynb) |
| Ablation (YOLOv8s 1280px) | [notebooks/advanced_extension_ablation.ipynb](../notebooks/advanced_extension_ablation.ipynb) |
| Advanced extension (RT-DETR) | [notebooks/advanced_extension_RTDETR.ipynb](../notebooks/advanced_extension_RTDETR.ipynb) |
| MIDI converter | [src/midi_converter.py](../src/midi_converter.py) |
| Data conversion utilities | [src/data_utils.py](../src/data_utils.py) |
| Gradio demo app | [src/app.py](../src/app.py) |
| Check-in 1 | [docs/check-in-1.md](check-in-1.md) |
| Check-in 2 | [docs/check-in-2.md](check-in-2.md) |
| Check-in 3 | [docs/check-in-3.md](check-in-3.md) |
| Model weights | [Google Drive](https://drive.google.com/drive/folders/1KHI4Ot9Y3CIJi-Ayn13PKKYbq34ZOTOq?usp=sharing) |
| Live demo | [Hugging Face Spaces](https://huggingface.co/spaces/samyu-vakkalanka/sheet-music-reader) |

---

## Citation

```bibtex
@dataset{tuggener2020deepscoresv2,
  title={DeepScoresV2},
  author={Tuggener, Lukas and Satyawan, Yvan Putra and Pacha, Alexander and
          Schmidhuber, Jürgen and Stadelmann, Thilo},
  year={2020},
  publisher={Zenodo},
  doi={10.5281/zenodo.4012193}
}
```