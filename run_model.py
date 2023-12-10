import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import numpy as np
import matplotlib.pyplot as plt
import cv2
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import gradio as gr

# Initialize model architecture with EfficientNet B3
model = EfficientNet.from_pretrained('efficientnet-b3')
num_classes = 2
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load("model_after_test.pth", map_location=torch.device('cpu')))
model.eval()

# Preparing image
def process_image(image):
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Function to predict and generate heatmap
def predict_and_visualize(image):
    img_tensor = process_image(image)

    # Make prediction
    logits = model(img_tensor)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][predicted_class].item() * 100  # Convert to percentage

    # Map the numeric prediction to a label
    class_labels = {0: "fake", 1: "real"}
    predicted_label = class_labels[predicted_class]

    # Grad-CAM
    target_layer = model._conv_head
    grad_cam = GradCAM(model, target_layer)
    mask, _ = grad_cam(img_tensor)
    heatmap, result = visualize_cam(mask, img_tensor)

    # Convert to displayable format
    result_image = np.transpose(result.numpy(), (1, 2, 0))
    result_image = np.clip(result_image, 0, 1)

    return result_image, f"{predicted_label} ({confidence:.2f}% confidence)"

# Gradio interface
iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="numpy"), "text"],
    title="Real vs Fake Face Detection",
    description="Upload an image to determine if the face is real or fake."
)

iface.launch()
