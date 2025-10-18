# Deepfake Image Detection with EfficientNet, FasterViT, and EfficientFormerV2

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
2. **Install dependencies:** We use Python 3.11+ (tested on 3.11/3.12).  
   Install with:
   ```bash
   pip install -r requirements.txt
