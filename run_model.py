import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from fastervit.models.faster_vit import FasterViT
from fastervit import create_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
import gradio as gr
import os

# Initialize EfficientNet model
efficientnet_model = EfficientNet.from_pretrained('efficientnet-b3')
num_classes = 2
in_features = efficientnet_model._fc.in_features
efficientnet_model._fc = nn.Linear(in_features, num_classes)
efficientnet_model.load_state_dict(torch.load("EfficientNetModel.pth", map_location=torch.device('cpu')))
efficientnet_model.eval()

# Initialize FasterViT model
model_name = 'faster_vit_0_224' 
faster_vit_model = create_model(model_name, pretrained=False, model_path=None)
num_classes = 2
in_features = faster_vit_model.head.in_features
faster_vit_model.head = nn.Linear(in_features, num_classes)
faster_vit_model.load_state_dict(torch.load("FasterVitModel.pth", map_location=torch.device('cpu')))
faster_vit_model.eval()

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
    class_labels = {0: "fake", 1: "real"}

    # EfficientNet Prediction
    logits_efficientnet = efficientnet_model(img_tensor)
    probabilities_efficientnet = F.softmax(logits_efficientnet, dim=1)
    predicted_class_efficientnet = torch.argmax(probabilities_efficientnet, dim=1).item()
    confidence_efficientnet = probabilities_efficientnet[0][predicted_class_efficientnet].item() * 100
    predicted_label_efficientnet = class_labels[predicted_class_efficientnet]

    # FasterViT Prediction
    logits_fastervit = faster_vit_model(img_tensor)
    probabilities_fastervit = F.softmax(logits_fastervit, dim=1)
    predicted_class_fastervit = torch.argmax(probabilities_fastervit, dim=1).item()
    confidence_fastervit = probabilities_fastervit[0][predicted_class_fastervit].item() * 100


    if predicted_class_fastervit in class_labels:
        predicted_label_fastervit = class_labels[predicted_class_fastervit]
    else:
        predicted_label_fastervit = f"Unknown Class: {predicted_class_fastervit}"


    # Grad-CAM for EfficientNet
    target_layer = efficientnet_model._conv_head
    grad_cam = GradCAM(efficientnet_model, target_layer)
    mask, _ = grad_cam(img_tensor)
    heatmap, result = visualize_cam(mask, img_tensor)

    # Convert to displayable format
    result_image = np.transpose(result.numpy(), (1, 2, 0))
    result_image = np.clip(result_image, 0, 1)

    # Combine results
    combined_result = f"EfficientNet: {predicted_label_efficientnet} ({confidence_efficientnet:.2f}% confidence)\n" \
                      f"FasterViT: {predicted_label_fastervit} ({confidence_fastervit:.2f}% confidence)"

    return result_image, combined_result

# Gradio interface
iface = gr.Interface(
    fn=predict_and_visualize,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="numpy"), "text"],
    title="Real vs Fake Face Detection",
    description="Upload an image to determine if the face is real or fake."
)

iface.launch()
