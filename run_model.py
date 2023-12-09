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

# Initialize model architecture with EfficientNet B3
model = EfficientNet.from_pretrained('efficientnet-b3')
num_classes = 2
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)
model.load_state_dict(torch.load("model_after_test.pth", map_location=torch.device('cpu')))
model.eval()

# Preparing image
def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(image).unsqueeze(0)

# Process the image
img_path = "Fake.jpg"
img_tensor = process_image(img_path)

# Make prediction
logits = model(img_tensor)
probabilities = F.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()

# Map the numeric prediction to a label
class_labels = {0: "real", 1: "fake"}
predicted_label = class_labels[predicted_class]

print(f"Predicted class: {predicted_label}")

# Grad-CAM
target_layer = model._conv_head
grad_cam = GradCAM(model, target_layer)
mask, _ = grad_cam(img_tensor)
heatmap, result = visualize_cam(mask, img_tensor)

# Display image
plt.imshow(np.transpose(result.numpy(), (1, 2, 0)))
plt.show()
