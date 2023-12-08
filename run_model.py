import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torch.nn.functional as F


# Step 1: Initialize your model architecture with EfficientNet b3
model = EfficientNet.from_pretrained('efficientnet-b3')

# Modify the final layer for binary classification
num_classes = 2  # genuine and deep fake
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, num_classes)

# Step 2: Load the saved model weights
model.load_state_dict(torch.load("model_after_test.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Step 3: Prepare your image
def process_image(image_path):
    image = Image.open(image_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    return image

# Replace 'Biden.jpg' with the path to the image you want to test
image = process_image("Biden.jpg")

# Step 4: Make a prediction
with torch.no_grad():
    prediction = model(image)
    # Process the prediction as needed
    logits = model(image)
    probabilities = F.softmax(logits, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    print(f"Raw logits: {logits}")
    print(f"Probabilities: {probabilities}")
    print(f"Predicted class: {predicted_class.item()}")

