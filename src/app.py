"""
app.py
Sheet Music Reader — Gradio Demo
"""

import gradio as gr
import numpy as np
from PIL import Image
from pathlib import Path
import tempfile
import subprocess
import sys
import os
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))
from ultralytics import YOLO, RTDETR
from midi_converter import convert_page_to_midi, ID_TO_CLASS

# ── Config ────────────────────────────────────────────────────────────────────

MODELS_DIR = Path(__file__).parent.parent / 'models'
FLUIDSYNTH_PATH = '/opt/homebrew/bin/fluidsynth'
SOUNDFONT_PATH = '/opt/homebrew/share/soundfonts/default.sf2'
TEMP_DIR = Path(tempfile.gettempdir()) / 'sheet_music_reader'
TEMP_DIR.mkdir(exist_ok=True)

AVAILABLE_MODELS = {
    'YOLOv8s — 640px (Check-in 2 Baseline)': MODELS_DIR / 'yolov8_640_best.pt',
    'YOLOv8s — 1280px (Ablation)': MODELS_DIR / 'yolov8_1280_best.pt',
    'RT-DETR-L — Transformer (Advanced)': MODELS_DIR / 'rtdetr_best.pt',
}

# ── Load models ───────────────────────────────────────────────────────────────

loaded_models = {}
for name, path in AVAILABLE_MODELS.items():
    if path.exists():
        print(f"Loading {name}...")
        if 'rtdetr' in str(path).lower():
            loaded_models[name] = RTDETR(str(path))
        else:
            loaded_models[name] = YOLO(str(path))
        print(f"  ✓ {name}")
    else:
        print(f"  ✗ {name} not found at {path}")

available_names = list(loaded_models.keys())
print(f"\n{len(available_names)} model(s) loaded.")

# ── Audio conversion ──────────────────────────────────────────────────────────

def midi_to_wav(midi_path, wav_path):
    cmd = [
        FLUIDSYNTH_PATH,
        '-ni',
        '-F', str(wav_path),
        SOUNDFONT_PATH,
        str(midi_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FluidSynth error: {result.stderr}")
    return wav_path

# ── Core pipeline ─────────────────────────────────────────────────────────────

def process_sheet_music(image, model_name, conf_threshold):
    if image is None:
        return None, None, "Please upload a sheet music image."
    if not loaded_models:
        return None, None, "No models loaded. Check that model files exist in models/."
    if model_name not in loaded_models:
        return None, None, f"Model '{model_name}' not loaded."

    try:
        model = loaded_models[model_name]

        img_path = TEMP_DIR / 'input.png'
        if isinstance(image, np.ndarray):
            Image.fromarray(image).save(img_path)
        else:
            image.save(img_path)

        # Detection
        results = model(str(img_path), conf=conf_threshold)
        annotated = results[0].plot(labels=False, conf=False)
        annotated_rgb = Image.fromarray(annotated[:, :, ::-1])

        boxes = results[0].boxes
        n_detections = len(boxes)
        class_counts = Counter(
            ID_TO_CLASS.get(int(c), 'unknown')
            for c in boxes.cls.cpu().numpy()
        )
        top_classes = ', '.join(
            f"{cls}: {cnt}"
            for cls, cnt in class_counts.most_common(5)
        )

        # MIDI conversion
        midi_path = TEMP_DIR / 'output.mid'
        result_path, key_info = convert_page_to_midi(
            image_path=str(img_path),
            yolo_results=results[0],
            output_path=str(midi_path),
            conf_threshold=conf_threshold
        )

        if result_path is None:
            return annotated_rgb, None, \
                f"Detected {n_detections} symbols but MIDI conversion failed — no staves found."

        # Audio
        audio_path = TEMP_DIR / 'output.wav'
        midi_to_wav(midi_path, audio_path)

        status = (
            f"✓ Detected {n_detections} symbols\n"
            f"Top classes: {top_classes}\n"
            f"Key signature: {key_info}\n"
            f"✓ MIDI conversion complete\n"
            f"✓ Audio ready"
        )

        return annotated_rgb, str(audio_path), status

    except Exception as e:
        import traceback
        return None, None, f"Error: {str(e)}\n{traceback.format_exc()}"

# ── Gradio UI ─────────────────────────────────────────────────────────────────

css = """
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Lato:ital,wght@0,300;0,400;1,300&display=swap');

/* ── CSS variable overrides — these beat the theme before Svelte touches anything ── */
:root,
.gradio-container {
    --body-background-fill:            #fdf7ef;
    --background-fill-primary:         #fdf7ef;
    --background-fill-secondary:       #fffbf4;
    --block-background-fill:           #fffbf4;
    --block-border-color:              #b87a8a;
    --block-border-width:              1.5px;
    --block-label-background-fill:     transparent;
    --block-label-border-color:        transparent;
    --block-label-text-color:          #5a7a5a;
    --block-title-text-color:          #5a7a5a;
    --input-background-fill:           #ffffff;
    --input-background-fill-focus:     #ffffff;
    --input-border-color:              #c9b08a;
    --input-border-color-focus:        #b87a8a;
    --input-placeholder-color:         #a08878;
    --color-accent:                    #b87a8a;
    --color-accent-soft:               #f5e6d8;
    --panel-background-fill:           #fffbf4;
    --panel-border-color:              #b87a8a;
    --link-text-color:                 #b87a8a;
    --link-text-color-hover:           #c99a6e;
    --checkbox-background-color:       #ffffff;
    --checkbox-border-color:           #b87a8a;
    --stat-background-fill:            #f5e6d8;
    --table-even-background-fill:      #fdf7ef;
    --table-odd-background-fill:       #fffbf4;
}

/* ── Page background ── */
html, body, #root, #root > div, .gradio-container, .app, footer {
    background-color: #fdf7ef !important;
}

.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}

/* ── Font base — broad to beat Svelte specificity ── */
.gradio-container p,
.gradio-container span,
.gradio-container div,
.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container button,
.gradio-container li {
    font-family: 'Lato', Georgia, sans-serif !important;
}

/* ── Serif for headings and labels ── */
.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container h4,
.gradio-container label,
.gradio-container label span,
.gradio-container .label-wrap span {
    font-family: 'Playfair Display', Georgia, serif !important;
}

.gradio-container h1 {
    color: #3d2314 !important;
    font-size: 2.3em !important;
    letter-spacing: 0.03em !important;
    font-weight: 600 !important;
}

/* Section headers — rose-to-gold gradient text */
.gradio-container h3 {
    background: linear-gradient(135deg, #b87a8a 0%, #c99a6e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: 0.04em !important;
    font-size: 1.15em !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid #c9b89a !important;
    margin-bottom: 12px !important;
}

/* ── Prose text ── */
.gradio-container p,
.gradio-container em,
.gradio-container strong {
    color: #4a3424 !important;
    line-height: 1.7 !important;
}

/* ── Component labels — sage green small caps ── */
.gradio-container label,
.gradio-container label span,
.gradio-container .label-wrap span {
    color: #5a7a5a !important;
    letter-spacing: 0.05em !important;
    font-size: 0.86em !important;
    text-transform: uppercase !important;
}

/* ── Input text & textarea — white, readable ── */
.gradio-container input:not([type="range"]),
.gradio-container textarea {
    background-color: #ffffff !important;
    color: #3d2314 !important;
}

/* ── Dropdown options panel ── */
.gradio-container .options,
.gradio-container ul.options {
    background-color: #fffbf4 !important;
    border: 1.5px solid #b87a8a !important;
    border-radius: 8px !important;
}
.gradio-container .options .item,
.gradio-container ul.options li {
    color: #3d2314 !important;
}
.gradio-container .options .item:hover,
.gradio-container ul.options li:hover {
    background-color: #f5e6d8 !important;
    color: #3d2314 !important;
}
.gradio-container .options .item.selected,
.gradio-container ul.options li.selected {
    background-color: #f0d8e0 !important;
    color: #3d2314 !important;
}

/* ── Image upload areas — warm parchment gold ── */
.gradio-container .upload-container,
.gradio-container .image-frame {
    background-color: #ede0c4 !important;
}
.gradio-container [data-testid="image"] .wrap {
    background-color: #ede0c4 !important;
    border: none !important;
}
.gradio-container .upload-container span,
.gradio-container .upload-container p,
.gradio-container [data-testid="image"] .wrap span,
.gradio-container [data-testid="image"] .wrap p {
    color: #5c4030 !important;
}

/* ── Slider — sage green thumb ── */
.gradio-container input[type="range"] {
    accent-color: #5a8a6a !important;
}

/* ── Run button — rose-to-gold pill ── */
.run-btn {
    background: linear-gradient(135deg, #b87a8a 0%, #c99a6e 100%) !important;
    border: none !important;
    border-radius: 30px !important;
    color: #fff8f2 !important;
    font-family: 'Playfair Display', Georgia, serif !important;
    font-size: 0.95em !important;
    letter-spacing: 0.1em !important;
    box-shadow: 0 3px 16px rgba(184, 122, 138, 0.35) !important;
    transition: box-shadow 0.2s ease, transform 0.2s ease !important;
}
.run-btn:hover {
    box-shadow: 0 5px 24px rgba(184, 122, 138, 0.55) !important;
    transform: translateY(-1px) !important;
}

/* ── Title block ── */
.title-block {
    text-align: center;
    padding: 32px 0 20px;
    border-bottom: 2px solid #c9b08a;
    margin-bottom: 8px;
}

/* ── Ornamental divider — sage green ── */
.ornament {
    text-align: center;
    color: #5a8a6a;
    font-size: 1.1em;
    letter-spacing: 0.4em;
    margin: 4px 0 20px;
}

/* ── Footer ── */
.footer-block {
    text-align: center;
    color: #5a7a5a;
    font-style: italic;
    font-family: 'Lato', sans-serif;
    font-size: 0.92em;
    padding: 10px 0 18px;
    border-top: 1px solid #c9b08a;
    margin-top: 12px;
}
.footer-block a { color: #b87a8a !important; }
"""

with gr.Blocks(
    title="Sheet Music Reader",
) as demo:

    gr.Markdown("""
<div class="title-block">

# Sheet Music Reader

Upload a page of **printed piano sheet music** to detect musical symbols,
convert to MIDI, and hear it played back.

Trained on DeepScores V2. Works best on clean printed scores. Handwritten music not supported.

</div>
<div class="ornament">◆ &nbsp; ◆ &nbsp; ◆</div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### The Score")
            image_input = gr.Image(
                label="Sheet Music Page",
                type="pil",
                height=400
            )
            model_selector = gr.Dropdown(
                choices=available_names,
                value='YOLOv8s — 1280px (Ablation)' if 'YOLOv8s — 1280px (Ablation)' in available_names else available_names[0],
                label="Detection Model"
            )
            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.3,
                step=0.05,
                label="Confidence Threshold"
            )
            run_btn = gr.Button(
                "Detect & Convert to Audio",
                variant="primary",
                elem_classes="run-btn"
            )

            gr.Examples(
                examples=[
                    [str(Path(__file__).parent / "examples/minuet.png")],
                    [str(Path(__file__).parent / "examples/amazing_grace.png")],
                    [str(Path(__file__).parent / "examples/canon_in_d.png")],
                ],
                inputs=image_input,
                label="Try an example"
            )
        with gr.Column(scale=1):
            gr.Markdown("### The Performance")
            annotated_output = gr.Image(
                label="Detected Symbols",
                height=400
            )
            audio_output = gr.Audio(
                label="Audio Playback",
                type="filepath"
            )
            status_output = gr.Textbox(
                label="Status",
                lines=4,
                interactive=False
            )

    run_btn.click(
        fn=process_sheet_music,
        inputs=[image_input, model_selector, conf_slider],
        outputs=[annotated_output, audio_output, status_output]
    )

    gr.Markdown("""
<div class="footer-block">

◆ &ensp; Georgetown DSAN-6500 Computer Vision &ensp; · &ensp; Dataset: <a href="https://zenodo.org/record/4012193">DeepScores V2</a> &ensp; · &ensp; Models: YOLOv8s, RT-DETR-L &ensp; ◆

</div>
    """)

if __name__ == '__main__':
    demo.launch(share=False, theme=gr.themes.Base(), css=css)
