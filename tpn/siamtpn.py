import torch
import torch.nn as nn
import numpy as np
from shuffleNet import Backbone
from PIL import Image
from torchvision import transforms



# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load and preprocess an image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image
        

class SiamTPN(torch.nn.Module):
    def __init__(self, backbone):
        super(SiamTPN, self).__init__()
        self.backbone = backbone
        
    
    def forward(self, x):
        x = self.backbone(x)
        return x
    


# Path to the input image
image_path = '../test/doggo.png'
image = load_image(image_path)

#Initializations
backbone = Backbone()

# Create the new model with the extracted stages
extracted_model = SiamTPN(backbone)

# Make a prediction and extract feature maps
with torch.no_grad():
    stage3_output, stage4_output, stage5_output = extracted_model(image)

# Print shapes of the extracted feature maps
print(f'Stage 3 output shape: {stage3_output.shape}')
print(f'Stage 4 output shape: {stage4_output.shape}')
print(f'Stage 5 output shape: {stage5_output.shape}')


