#!/usr/bin/env python3
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from siamtpn import SiamTPN
from shuffleNet import Backbone
from head import ClassificationHead, RegressionHead
from poolingAttention import PABlock
from depth import DepthwiseCorrelation
import matplotlib.pyplot as plt
import torch.nn.functional as F


# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
transform2 = transforms.Compose([
    transforms.Resize(80),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SiamTPN(nn.Module):
    def __init__(self, d_model=464, num_heads=8, pooling_size=2, pooling_stride=2, padding=7):
        super(SiamTPN, self).__init__()
        self.backbone = Backbone()
        self.tpn = PABlock(d_model=d_model, num_heads=num_heads, pooling_size=pooling_size, pooling_stride=pooling_stride)
        self.depthwise_corr = DepthwiseCorrelation(padding=padding)
        self.classification_head = ClassificationHead(in_channels=d_model, num_classes=1)
        self.regression_head = RegressionHead(in_channels=d_model, num_coords=4)

    def forward(self, template, search):
        # Extract features from template and search images
        P3_template, P4_template, P5_template = self.backbone(template)
        P3_search, P4_search, P5_search = self.backbone(search)

        # Process features through the TPN
        template_features = self.tpn(P3_template, P4_template, P5_template)
        #print("Temprale features,",template_features.shape )
        search_features = self.tpn(P3_search, P4_search, P5_search)

    
        # Perform depth-wise correlation
        correlation_maps = self.depthwise_corr(template_features, search_features)

        # Classification and regression
        classification_output = self.classification_head(correlation_maps)
         # Resize the matrix to [224, 224] using interpolation and Remove the batch and channel dimensions
        classification_output = F.interpolate(classification_output, size=(224, 224), mode='bilinear', align_corners=False)
        classification_output = classification_output.squeeze()


        regression_output = self.regression_head(correlation_maps)

        return classification_output, regression_output
    
# Path to your .pth file
model_weights_path = 'siamtpn_model2.pth'

# Initialize the SiamTPN model
model = SiamTPN(num_heads=8, pooling_size=2, pooling_stride=2, padding=7).to(device)

# Load the state dictionary
state_dict = torch.load(model_weights_path, map_location=device)

# Load the weights into the model
model.load_state_dict(state_dict)

model.eval()


# Load the image
image = Image.open("../dataset_1vid/turtle/img/00001507.jpg")
image = transform(image)
image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device

print(image.shape)


# Load the image
template = Image.open("../dataset_1vid/turtle/img/00000001.jpg")

x, y, w, h = 578, 163, 241, 159
print("x, y, w, h", x, y, w, h)
d = max(w, h)
w = h = d

template_crop = template.crop((x, y, x + w, y + h))
template_crop = transform2(template_crop)
template_crop = template_crop.unsqueeze(0).to(device)

print("template shape", template_crop.shape)

classification_output, regression_output = model(template_crop, image)

print(classification_output)

 # Threshold the model's output to get binary prediction
binary_output = (classification_output > 0.7).float()

print(binary_output)

# Move the tensor to the CPU
binary_output = binary_output.cpu()
# Convert the tensor to a NumPy array
image_np = binary_output.numpy()

# Display the image using Matplotlib
plt.imshow(image_np, cmap='gray')  # Use cmap='gray' for grayscale image
plt.title('Tensor Image')
plt.axis('off')  # Turn off axis
plt.show()