# Sheet Music Reader

A computer vision project that detects and classifies musical symbols in printed sheet music images, with the goal of converting sheet music into playable output.

Built as a semester-long CV project using the [DeepScores V2](https://zenodo.org/records/4012193) dataset.

---

## Navigation

| Phase | Document |
|-------|----------|
| Check-in 1 — Problem Framing & Data | [check-in-1.md](docs/check-in-1.md) |
| Check-in 2 — Fundamentals | [check-in-2.md](docs/check-in-2.md) |
| Check-in 3 — Advanced Extension  | [check-in-3.md](docs/check-in-3.md) |

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
├── data/
├── models/
└── src/
│   └── data_utils.py
│   └── midi_converter.py
```

## Model Weights
Model weights are not stored in this repo due to file size.

---

## Dataset Access

This project uses the DeepScores V2 dataset. See [check-in-1.md](docs/check-in-1.md) for full access and download instructions.

*Requirements file will be added as the project develops.*