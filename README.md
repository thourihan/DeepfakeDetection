# CSC 461 Final: Deepfake Image Detection with EfficientNet and FasterViT

## Project Overview
In this project, we used EfficientNet and FasterViT to detect deepfake images from real ones. Deepfakes pose significant challenges in the realms of security, privacy, and information integrity. Our goal was to develop a robust model capable of accurately distinguishing between real and deep fake images by using the latest advancements in machine learning and image processing.

Checkout the research paper [here](https://docs.google.com/document/d/1Duh0sKFxBPB-_-t1U8HRtR3py7OVKb715McTbgyHh7Q/edit?usp=sharing)!

You can also find some runs on WandB, we are hoping to implement a more excessive hyperparameter sweep feature in the future: https://wandb.ai/461

## Features
- **EfficientNet and FasterViT Models:** Utilizes pre-trained EfficientNet and FasterViT models to classify images as "real" or "fake".
- **Gradio Interface:** Provides an easy-to-use web interface for uploading images and viewing predictions.
- **Heatmap Visualization:** Generates heatmaps for each image to show what the model is "looking at" using Grad-CAM.

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
1. Clone the Repository
2. Install dependencies:
Make sure you have Python installed on your system, we used Python 3.11.6. Install the required packages using `pip install -r requirements.txt`
3. Train the models:
We have provided pre-trained model files for your convenience. You can download our EfficientNetModel.pth and FasterVitModel.pth files from the following Google Drive links:
    - EfficientNetModel.pth: [Download Here](https://drive.google.com/file/d/1xVW50FY02utzv_ux-474tNXU8d7giKkD/view?usp=sharing)
    - FasterVitModel.pth: [Download Here](https://drive.google.com/file/d/120Lz6ueJEPzhTHkxA58kmwtU6IY6O6NX/view?usp=sharing)
    - Note: If you prefer to train the models yourself, you can use the train_model.ipynb (for EfficientNet) and train_model_fastervit.ipynb (for FasterViT).
4. Run the Application:
To start the Gradio web interface, run `python run_model.py`

## Usage
After launching the application, navigate to the Gradio web URL displayed in your terminal.
1. **Upload an Image:** Click the upload button and select an image with a face.
2. **View Predictions:** The model will process the image and display whether the face is real or fake, along with confidence scores.
3. **Heatmap Visualization:** Alongside the predictions, a heatmap overlay on the image will be displayed, showing the areas most influential in the model's decision.

## Dataset
The models are trained on a dataset available on Kaggle: [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images). Make sure you download and place the dataset in the appropriate directory before running the training notebooks.

