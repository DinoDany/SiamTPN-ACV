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

class SingleVideoDataset(Dataset):
    def __init__(self, video_dir, frame_list_file, transform=None, transform2=None):
        self.video_dir = video_dir
        self.transform = transform
        self.transform2 = transform2
        self.frame_list, self.annotations = self.load_data(frame_list_file)

    def load_data(self, frame_list_file):
        template_search_pairs = []
        annotations = []

        video_path = os.path.join(self.video_dir, 'img')
        groundtruth_file = os.path.join(self.video_dir, 'groundtruth.txt')

        # Read frame names from the list file
        with open(frame_list_file, 'r') as f:
            frame_names = f.read().splitlines()
        #print("frame_names", frame_names)

        # Read ground truth annotations
        with open(groundtruth_file, 'r') as f:
            annots = f.readlines()

        # Create template from the first frame and its bounding box
        template_frame = Image.open(os.path.join(video_path, frame_names[0])).convert('RGB')
        x, y, w, h = [int(x) for x in annots[0].strip().split(',')]
        #print("x, y, w, h", x, y, w, h)
        d = min(w, h)
        w = h = d
        template_crop = template_frame.crop((x, y, x + w, y + h))
        #print(template_crop.size)

        # Map frame names to full paths and annotations
        for frame_name in frame_names:
            frame_file = os.path.join(video_path, frame_name)
            #print(frame_file)
            frame_index = int(frame_name.split('.')[0]) - 1  # assuming frame names are like '0001.jpg', '0002.jpg', etc.
            search_frame = Image.open(frame_file).convert('RGB')
            search_bbox = [int(x) for x in annots[frame_index].strip().split(',')]
            template_search_pairs.append((template_crop, search_frame))
            annotations.append(search_bbox)

        #print("annotation", annotations)
        #print("annotation", len(annotations))

        return template_search_pairs, annotations

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        #print("index: ", idx)
        #print("Get item is running")
        (template_frame, search_frame) = self.frame_list[idx]
        frame_annotation = self.annotations[idx]
        if self.transform:
            template_frame = self.transform2(template_frame)
            search_frame = self.transform(search_frame)

        x_fact = 224/ 1280
        y_fact = 224 / 720

        #Re scale the annotations to be 224x224
        annotation = int(frame_annotation[0]* x_fact), int(frame_annotation[1] * y_fact), int(frame_annotation[2]* x_fact), int(frame_annotation[3] * y_fact)
        #print("New annotation", annotation)

        return template_frame, search_frame, annotation

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Paths to the video directory and frame list files
video_dir = '../dataset_1vid/turtle'  # Replace with the actual path to your video directory
testing_list_file = '../dataset_1vid/turtle/testing.txt'

# Load the single video dataset for training and testing

test_dataset = SingleVideoDataset(video_dir=video_dir, frame_list_file=testing_list_file, transform=transform, transform2=transform2)
print(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)






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
model_weights_path = 'siamtpn_model.pth'

# Initialize the SiamTPN model
model = SiamTPN(num_heads=8, pooling_size=2, pooling_stride=2, padding=7).to(device)

# Define loss functions and optimizer
classification_criterion = nn.CrossEntropyLoss()

# Load the state dictionary
state_dict = torch.load(model_weights_path, map_location=device)

# Load the weights into the model
model.load_state_dict(state_dict)


# Function to test the model
def test(model, device, test_loader, criterion):
    model.eval()  # Set the model to evaluation mode
    total_classification_loss = 0.0
    total_correct = 0
    total_pixels = 0

    with torch.no_grad():  # Disable gradient calculation
        for batch_idx, (template_frame, search_frame, anno) in enumerate(test_loader):
            # Move data to the same device as the model

            template_frame = template_frame.to(device)
            search_frame = search_frame.to(device)
            anno = [a.to(device) for a in anno]  # Ensure annotation is on the correct device

            # Forward pass
            classification_output, regression_output = model(template_frame, search_frame)

            # Extract annotation values
            x, y, w, h = anno[0][0], anno[1][0], anno[2][0], anno[3][0]

            # Create a matrix of zeros
            classification_labels = torch.zeros((224, 224), device=device)
            classification_labels[y:y+h, x:x+w] = 1

            # Compute classification loss
            classification_loss = classification_criterion(classification_output, classification_labels)
            print(f"Batch {batch_idx}: Classification Loss: {classification_loss.item()}")

            total_classification_loss += classification_loss.item()

            # Threshold the model's output to get binary predictions
            binary_output = (classification_output > 0.5).float()

            # Calculate the number of correct predictions
            correct = (binary_output == classification_labels).sum().item()
            total_correct += correct
            total_pixels += classification_labels.numel()

            # Calculate and print batch accuracy
            batch_accuracy = (correct / classification_labels.numel()) * 100
            print(f"Batch {batch_idx}: Accuracy: {batch_accuracy:.2f}%")

    # Calculate and print epoch accuracy
    epoch_accuracy = (total_correct / total_pixels) * 100
    epoch_loss = total_classification_loss / len(test_loader)
    print(f"Test set: Average loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

# Test the model
test(model, device, test_loader, classification_criterion)



