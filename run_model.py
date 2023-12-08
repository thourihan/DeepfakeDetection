import torch
import torch.nn as nn 
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import numpy as np
import matplotlib.pyplot as plt
import cv2 


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

# Grad-CAM
class GradCam:
    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self.gradient = None
        self.register_hooks()

    def save_gradient(self, grad):
        self.gradient = grad

    def register_hooks(self):
        layer = getattr(self.model, self.layer_name)

        def forward_hook(module, input, output):
            self.feature = output

        def backward_hook(module, grad_in, grad_out):
            self.save_gradient(grad_out[0])

        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def generate_cam(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        model_output.backward(gradient=one_hot_output, retain_graph=True)

        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))[..., np.newaxis, np.newaxis]
        feature = self.feature[0].cpu().data.numpy()

        cam = np.sum(weight * feature, axis=0)
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam

def show_cam_on_image(img, mask, alpha=0.6):
    # Resize mask to match the image size
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255

    # Blend heatmap and image
    cam = (1 - alpha) * np.float32(img) + alpha * heatmap
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


# Process the image
img_path = "Biden.jpg"
img = np.array(Image.open(img_path).convert('RGB'))
img_tensor = process_image(img_path)

# Grad-CAM
grad_cam = GradCam(model, '_conv_head')
target_class = torch.argmax(model(img_tensor)).item()
cam = grad_cam.generate_cam(img_tensor, target_class)
cam_image = show_cam_on_image(img / 255.0, cam)

# Make prediction
logits = model(img_tensor)
probabilities = F.softmax(logits, dim=1)
predicted_class = torch.argmax(probabilities, dim=1).item()

# Map the numeric prediction to a label
class_labels = {0: "real", 1: "fake"}
predicted_label = class_labels[predicted_class]

print(f"Predicted class: {predicted_label}")

# Display image
plt.imshow(cam_image)
plt.show()
