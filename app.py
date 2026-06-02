# app.py
# ── DO NOT import torch ──────────────────────────────────────────────────────
# All predictions are pre-computed. This app only reads numpy files and renders
# matplotlib figures. No model inference happens here.
# ────────────────────────────────────────────────────────────────────────────

import numpy as np
import gradio as gr
from loader import load_demo_data
from visualizer import make_heatmap, compute_stats

# ── Load data once at startup ─────────────────────────────────────────────────
print("Loading demo data...")
DATA         = load_demo_data()
PREDS        = DATA["preds"]          # (22, 140, 124, 16)
INPUTS       = DATA["inputs"]         # (22, 10, 140, 124)
LAT          = DATA["lat"]            # (140, 124) or None
LON          = DATA["lon"]            # (140, 124) or None
SAMPLE_IDX   = DATA["sample_indices"] # (22,)
N_WINDOWS    = PREDS.shape[0]         # 22
N_HOURS      = PREDS.shape[3]         # 16
print(f"Ready — {N_WINDOWS} windows, {N_HOURS} forecast hours.")
# ─────────────────────────────────────────────────────────────────────────────


def update(window_slider: int, hour_slider: int):
    """
    Called whenever either slider changes.
    Returns: (input_image, pred_image, stats_markdown)
    """
    w = int(window_slider)
    h = int(hour_slider) - 1          # convert 1-indexed UI → 0-indexed array

    # (140, 124) arrays
    input_frame = INPUTS[w, -1]       # last of the 10 input hours
    pred_frame  = PREDS[w, :, :, h]   # predicted PM2.5 at hour h+1

    # Shared color scale so the two heatmaps are directly comparable
    combined_max = max(float(input_frame.max()), float(pred_frame.max()))
    vmin = 0.0
    vmax = min(max(combined_max * 1.1, 50.0), 300.0)  # at least 0–50 scale

    original_window = int(SAMPLE_IDX[w])

    input_img = make_heatmap(
        input_frame,
        title=f"Input PM2.5 — Last Known Hour\n(Test window {original_window})",
        vmin=vmin, vmax=vmax,
        lat=LAT, lon=LON,
    )

    pred_img = make_heatmap(
        pred_frame,
        title=f"Predicted PM2.5 — +{h + 1}h Forecast\n(Test window {original_window})",
        vmin=vmin, vmax=vmax,
        lat=LAT, lon=LON,
    )

    stats = compute_stats(pred_frame, input_frame)
    stats_md = "\n".join(f"**{k}:** {v}" for k, v in stats.items())

    return input_img, pred_img, stats_md


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="PM2.5 Pollution Forecast",
    theme=gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
    ),
    css="""
    .gradio-container { max-width: 1100px; margin: auto; }
    h1 { text-align: center; }
    .subtitle { text-align: center; color: #888; font-size: 0.9em; margin-top: -8px; }
    """,
) as demo:

    gr.Markdown("# 🌬️ PM2.5 Pollution Forecasting")
    gr.HTML('<p class="subtitle">ANRF AISEHack Phase 2 — Deep Learning Air Quality Forecast over India</p>')

    with gr.Row():
        with gr.Column(scale=1):
            window_slider = gr.Slider(
                minimum=0,
                maximum=N_WINDOWS - 1,
                step=1,
                value=0,
                label=f"Test Window  (0 – {N_WINDOWS - 1})",
                info="Each window is a different time period from the test set.",
            )
        with gr.Column(scale=1):
            hour_slider = gr.Slider(
                minimum=1,
                maximum=N_HOURS,
                step=1,
                value=1,
                label=f"Forecast Hour  (+1h – +{N_HOURS}h)",
                info="How many hours ahead the model is predicting.",
            )

    with gr.Row():
        input_img = gr.Image(
            label="Input PM2.5 — Last Known Hour",
            type="pil",
            height=380,
        )
        pred_img = gr.Image(
            label="Predicted PM2.5",
            type="pil",
            height=380,
        )

    stats_box = gr.Markdown(label="Forecast Statistics")

    gr.Markdown("""
---
**Model:** ConvLSTM encoder + Fourier Neural Operator (FNO) hybrid  
**Training:** Kaggle T4 GPU · ANRF competition dataset · 4 months × 16 atmospheric features  
**Grid:** 140 × 124 spatial points · Northern India  
**Input:** 10 hours of atmospheric data → **Output:** 16-hour PM2.5 forecast  
**Competition Rank:** 24 · Final Score: 21.244 (sMAPE-based)
""")

    # ── Wire up callbacks ────────────────────────────────────────────────────
    inputs_list  = [window_slider, hour_slider]
    outputs_list = [input_img, pred_img, stats_box]

    window_slider.change(fn=update, inputs=inputs_list, outputs=outputs_list)
    hour_slider.change(fn=update,   inputs=inputs_list, outputs=outputs_list)

    # Load initial view on page open
    demo.load(fn=lambda: update(0, 1), outputs=outputs_list)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
