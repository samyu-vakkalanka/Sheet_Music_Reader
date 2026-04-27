# Check-in 3: Advanced Extension

## Overview
This check-in extends the YOLOv8s baseline from Check-in 2 in two ways:
1. **Advanced extension:** RT-DETR, a transformer-based object detector, replacing the CNN backbone with an attention-based architecture
2. **Ablation:** YOLOv8s trained at higher resolution (imgsz=1280) to directly address the stem/ledgerLine zero-detection failure identified in Check-in 2

See the full notebook: [`notebooks/advanced_extension.ipynb`](../notebooks/advanced_extension.ipynb)

---

## 1. Advanced Extension: RT-DETR (Transformer-Based Detection)

### Motivation
The main problem found in Check-in 2 was complete zero-detection on stem, ledgerLine, and augmentationDot. These are all thin/small symbols that disappear when high-resolution sheet music is downscaled to 640px. Another motivation was trying to compare the purely CNN approach in YOLOv8 with RT-DETR which has a transformer-based encoder (AIFI = Attention-based Intra-scale Feature 
Interaction), which may better capture global context and long-range dependencies between symbols.

### Architecture
RT-DETR-L uses:
- HGNet backbone (hybrid CNN feature extractor)
- AIFI transformer encoder for multi-scale feature interaction
- RT-DETR decoder with learned object queries

Key difference from YOLOv8: the AIFI module applies self-attention across spatial positions at the highest feature map level, allowing the model to reason about relationships between distant symbols. This could potentially help with context-dependent 
classes like `accidentalFlat` vs `keyFlat`.

Model size: 32.8M parameters, 108.1 GFLOPs (vs YOLOv8s: 11.1M params, 28.7 GFLOPs)

### Training Configuration
- Model: RT-DETR-L (pretrained on COCO)
- Image size: 640px
- Batch size: 4
- Epochs: 50 (patience=10)
- Hardware: Tesla T4 (Google Colab)
- Dataset: DeepScores V2 dense subset (same as Check-in 2)

### Results
<!-- To be filled in after training -->

---

## 2. Ablation: YOLOv8s at Higher Resolution (imgsz=1280)

### Motivation
The stem/ledgerLine zero-detection failure in Check-in 2 was directly caused by downscaling 1960×2772px sheet music pages to 640px for inference. Stems that are 3-4px wide at full resolution become less than a pixel wide and invisible. Training and inferring at 1280px might help save some of the more fine-grained detail.

### Training Configuration
- Model: YOLOv8s (same architecture as Check-in 2 baseline)
- Image size: 1280px (vs 640px in baseline)
- Batch size: 4 (reduced from 8 due to memory)
- Epochs: 50 (patience=10)
- Everything else identical to Check-in 2

### Results
<!-- To be filled in after training -->

---

## 3. Comparison

### Metrics Summary

| Model | imgsz | mAP@0.5 | mAP@0.5:0.95 | stem AP | ledgerLine AP | augDot AP |
|---|---|---|---|---|---|---|
| YOLOv8s (Check-in 2) | 640 | 0.489 | 0.275 | 0.000 | 0.000 | 0.000 |
| YOLOv8s (ablation) | 1280 | coming soon | coming soon | coming soon | coming soon | coming soon |
| RT-DETR-L (advanced) | 640 | coming soon | coming soon | coming soon | coming soon | coming soon |

### Discussion
<!-- To be filled in after training -->

---

## 4. Failure Analysis

### RT-DETR
<!-- To be filled in after training -->

### YOLOv8s at 1280px
<!-- To be filled in after training -->

---

## 5. End-to-End Demo
The full pipeline demo is available as a local Gradio app in `app.py`.

**To run:**
```bash
pip install ultralytics gradio music21 midi2audio
brew install fluidsynth  # Mac only
python app.py
```

Then upload any printed sheet music image and the app will detect symbols, convert to MIDI, and play audio in the browser.

---

## 6. Plan for Final Deliverable

**Remaining work:**
- Polish demo app
- Final report write-up combining all three check-ins into a coherent narrative
- Presentation slides

**Highest priority next steps:**
- Improve pitch calibration in midi_converter.py

**Known risks:**
- RT-DETR may not outperform YOLOv8s on this dataset despite larger size
- 1280px training may still not detect stems if the issue is annotation quality rather than resolution
- Gradio app requires local fluidsynth installation which may complicate the demo setup