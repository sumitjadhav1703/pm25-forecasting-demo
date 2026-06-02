---
title: PM2.5 Pollution Forecast
emoji: 🌬️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# 🌬️ PM2.5 Pollution Forecasting Demo

**ANRF AISEHack Phase 2 — Theme 2 — Pollution Forecasting (IIT Delhi)**

This demo visualizes predictions from a **ConvLSTM + Fourier Neural Operator (FNO)** hybrid model
trained to forecast PM2.5 air pollution levels across a 140×124 spatial grid over Northern India.

## How to Use

1. Use the **Test Window** slider to select a time period from the test dataset
2. Use the **Forecast Hour** slider to select how far ahead (+1h to +16h)
3. Compare the last known PM2.5 map (left) with the model's forecast (right)
4. Read the statistics below the maps

## Model Architecture

| Component | Details |
|-----------|---------|
| Encoder   | Stacked ConvLSTM (2 layers) |
| Spatial   | Fourier Neural Operator (FNO) |
| Decoder   | UNet with SE blocks |
| Input     | 10 hours × 20 atmospheric features × 140×124 grid |
| Output    | 16-hour PM2.5 forecast |
| Training  | Kaggle T4 GPU, ~8 hours |

## Competition Results

- **Competition:** ANRF AISEHack Phase 2 — Theme 2 (IIT Delhi)  
- **Team:** MGM  
- **Final Rank:** 24  
- **Final Score:** 21.244 (sMAPE-based metric)  
- Improved from rank 33 (score 56.232) → rank 24 (score 21.244)

## Dataset

The competition dataset contains 4 months of WRF-simulated atmospheric data:
APRIL_16, JULY_16, OCT_16, DEC_16. Features include PM2.5, wind components,
temperature, PBLH, and various emission tracers.

> Built by Sumit — B.Tech AI & Data Science, JNEC MGM University
