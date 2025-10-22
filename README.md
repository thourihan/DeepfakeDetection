# Deepfake Image Detection with EfficientNet, FasterViT, and EfficientFormerV2

![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![License](https://img.shields.io/github/license/thourihan/DeepfakeDetection.svg)
![GitHub release](https://img.shields.io/github/v/release/thourihan/DeepfakeDetection.svg)
![Last Commit](https://img.shields.io/github/last-commit/thourihan/DeepfakeDetection.svg)
![Stars](https://img.shields.io/github/stars/thourihan/DeepfakeDetection?style=social)
![Framework](https://img.shields.io/badge/framework-PyTorch-red.svg)

## Project Overview
This project detects deepfake images using modern CNN/ViT-family backbones. We use EfficientNet, FasterViT, and EfficientFormerV2-S1 to balance speed and accuracy, and to diversify architectures to reduce brittleness to dataset-specific artifacts. Models are integrated via `timm`, with reproducible training/evaluation and Grad-CAM visualizations.

Check out the research paper [here](https://docs.google.com/document/d/1Duh0sKFxBPB-_-t1U8HRtR3py7OVKb715McTbgyHh7Q/edit?usp=sharing)!

## Features
- **Multiple Backbones (via timm):** EfficientNet, FasterViT, and EfficientFormerV2-S1 for complementary accuracy/latency trade-offs.
- **Gradio Interface:** Simple web UI for uploading images and viewing model predictions.
- **Grad-CAM Visualization:** Heatmaps highlighting the regions driving model decisions (available across models).
- **Training & Evaluation Scripts:** Ssripts for training and evaluation with Grad-CAM oututs. 

## Sample Predictions and Heatmap Visualizations
Here are a couple of examples of the model's output, along with heatmap visualizations for each.

### Deepfake Detection Example
**Mark Zuckerberg Deepfake**  
Predictions:  
EfficientNet: fake (62.11% confidence)  
FasterViT: fake (97.33% confidence)

<table>
  <tr>
    <td>
      <p><strong>Input Image</strong></p>
      <img src="images/mark-zuckerberg-deepfake.webp" alt="Mark Zuckerberg Deepfake" width="361" height="224"/>
    </td>
    <td>
      <p><strong>Heatmap Visualization</strong></p>
      <img src="images/mark-zuckerberg-deepfake-heatmap.png" alt="Heatmap of Mark Zuckerberg Deepfake" width="224" height="224"/>
    </td>
  </tr>
</table>

### Real Image Detection Example
**Donald Trump Real**  
Predictions:  
EfficientNet: real (97.83% confidence)  
FasterViT: real (98.55% confidence)

<table>
  <tr>
    <td>
      <p><strong>Input Image</strong></p>
      <img src="images/donald-trump-real.jpg" alt="Donald Trump Real" width="301" height="224"/>
    </td>
    <td>
      <p><strong>Heatmap Visualization</strong></p>
      <img src="images/donald-trump-real-heatmap.png" alt="Heatmap of Donald Trump Real" width="224" height="224"/>
    </td>
  </tr>
</table>

## Installation
To set up the project, follow these steps:
1. **Clone the repository.**
2. **Install dependencies:** We use Python 3.12+
   Install with:
   ```bash
   pip install -r requirements.txt
```

## Orchestrating runs (training & inference)

The new `orchestrator.py` script provides a thin automation layer for
deepfake frame classification experiments. It reads the YAML file at
`config/run.yaml`, spins up per-model run directories under
`runs/{model}/{timestamp}/`, and wires environment variables so the
existing training scripts can resume, pick seeds, and respect custom
dataset splits.

1. **Train models listed in the config** (ensure `mode: training`):
   ```bash
   python orchestrator.py --config config/run.yaml
   ```
2. **Evaluate models headlessly** (toggle `mode: inference` in the same
   config to reuse the dataset/weights settings):
   ```bash
   python orchestrator.py --config config/run.yaml
   ```

Each run directory contains `checkpoints/` (latest & best checkpoints),
`logs/` (console output plus JSONL metrics), and `plots/` (confusion
matrix and ROC curve when labels are available). The setup targets
frame-level deepfake vs. real classification but works for multiclass
ImageFolder datasets as well (e.g., MNIST variants converted to RGB).
