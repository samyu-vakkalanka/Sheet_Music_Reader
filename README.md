# Sheet Music Reader

A computer vision project that detects and classifies musical symbols in printed sheet music images, with the goal of converting sheet music into playable output.

Built as a semester-long CV project using the [DeepScores V2](https://zenodo.org/records/4012193) dataset.

**Key Results:**
- YOLOv8s at 640px: mAP@0.5 = 0.489
- YOLOv8s at 1280px: mAP@0.5 = 0.594 (+21%)
- RT-DETR-L: mAP@0.5 = 0.281
- End-to-end: sheet music image → playable audio

---

## Navigation

| Phase | Document |
|-------|----------|
| Check-in 1 — Problem Framing & Data | [check-in-1.md](docs/check-in-1.md) |
| Check-in 2 — Fundamentals | [check-in-2.md](docs/check-in-2.md) |
| Check-in 3 — Advanced Extension  | [check-in-3.md](docs/check-in-3.md) |
| Final Report | [final.md](docs/final.md) |

---

## Project Structure
```
sheet-music-reader/
├── README.md
├── docs/
│   └── check-in-1.md
│   └── check-in-2.md
├── notebooks/
│   └── eda.ipynb
│   └── eda_outputs/
│       └── class_distribution.png
│       └── sample_1.png
│       └── sample_2.png
│       └── sample_3.png
│   └── initial_baseline.ipynb
│   └── initial_baseline_outputs/
│       └── identification.png
│       └── thresholds.png
│   └── classical_baseline.ipynb
│   └── classical_baseline_outputs/
│       └── hog_confusion_matrix.png
│       └── hog_confusion_pairs.png
│       └── hog_failure_cases.png
│       └── hog_sample_crops.png
│   └── cnn_baseline.ipynb
│   └── cnn_baseline_outputs/
│       └── test3.png
│       └── yolo_map_per_class.png
│       └── yolo_predictions.png
│       └── yolo_resolution_failure.png
│   └── advanced_extension_ablation.ipynb
│   └── advanced_extension_ablation_outputs/
│       └── 1280_confusion_matrix_normalized.png
│       └── 1280_results.png
│       └── stem_failure_1280.png
│       └── yolo_1280_comparison.png
│       └── yolo_1280_predictions.png
│   └── advanced_extension_RTDETR.ipynb
│   └── advanced_extension_RTDETR_outputs/
│       └── confusion_matrix_normalized.png
│       └── demo.png
│       └── results.png
│       └── three_way_comparison.png
├── data/
├── models/
└── src/
│   └── app.py
│   └── data_utils.py
│   └── midi_converter.py
```

## Model Weights
Model weights are not stored in this repo due to file size. Download from [Google Drive](https://drive.google.com/drive/folders/1KHI4Ot9Y3CIJi-Ayn13PKKYbq34ZOTOq?usp=sharing).

---

## Dataset Access

This project uses the DeepScores V2 dataset. See [check-in-1.md](docs/check-in-1.md) for full access and download instructions.

## Setup

```bash
# Clone the repo
git clone https://github.com/samyu-vakkalanka/Sheet_Music_Reader.git
cd Sheet_Music_Reader

# Install dependencies
pip install -r requirements.txt

# Install fluidsynth (Mac only, for local audio playback)
brew install fluidsynth

# Download model weights from Google Drive (link above)
# Place in models/ directory
```

## Demo

A working demo with the 2 YOLOv8s models is available on [Hugging Face](https://huggingface.co/spaces/samyu-vakkalanka/sheet-music-reader)

Alternatively, a local Gradio app with all 3 models is available for live inference and audio playback.

**Requirements:**
```bash
pip install ultralytics gradio music21 midi2audio
brew install fluidsynth  # Mac only
```

**Run:**
```bash
python src/app.py
```

Then open `http://127.0.0.1:7860` in your browser. Upload any printed piano sheet music page and the app will detect symbols, convert to MIDI, and play audio.

Model weights are not included in this repo. Download from [Google Drive](https://drive.google.com/drive/folders/1KHI4Ot9Y3CIJi-Ayn13PKKYbq34ZOTOq?usp=sharing) and place in `models/`.
