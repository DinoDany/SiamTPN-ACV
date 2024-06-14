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

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SingleVideoDataset(Dataset):
    def __init__(self, video_dir, frame_list_file, transform=None):
        self.video_dir = video_dir
        self.transform = transform
        self.frame_list, self.annotations = self.load_data(frame_list_file)

    def load_data(self, frame_list_file):
        template_search_pairs = []
        annotations = []

        video_path = os.path.join(self.video_dir, 'img')
        groundtruth_file = os.path.join(self.video_dir, 'groundtruth.txt')

        # Read frame names from the list file
        with open(frame_list_file, 'r') as f:
            frame_names = f.read().splitlines()

        # Read ground truth annotations
        with open(groundtruth_file, 'r') as f:
            annots = f.readlines()

        # Create template from the first frame and its bounding box
        template_frame = Image.open(os.path.join(video_path, frame_names[0])).convert('RGB')
        x, y, w, h = [int(x) for x in annots[0].strip().split(',')]
        template_crop = template_frame.crop((x, y, x + w, y + h))

        # Map frame names to full paths and annotations
        for frame_name in frame_names:
            frame_file = os.path.join(video_path, frame_name)
            frame_index = int(frame_name.split('.')[0]) - 1  # assuming frame names are like '0001.jpg', '0002.jpg', etc.
            search_frame = Image.open(frame_file).convert('RGB')
            search_bbox = [int(x) for x in annots[frame_index].strip().split(',')]
            template_search_pairs.append((template_crop, search_frame))
            annotations.append(search_bbox)

        return template_search_pairs, annotations

    def __len__(self):
        return len(self.frame_list)

    def __getitem__(self, idx):
        frame_file = self.frame_list[idx]
        frame_annotation = self.annotations[idx]

        frame = Image.open(frame_file).convert('RGB')
        if self.transform:
            frame = self.transform(frame)

        return frame, frame_annotation

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Paths to the video directory and frame list files
video_dir = '../dataset_1vid/turtle'  # Replace with the actual path to your video directory
training_list_file = '../dataset_1vid/turtle/training.txt'
testing_list_file = '../dataset_1vid/turtle/testing.txt'

# Load the single video dataset for training and testing
train_dataset = SingleVideoDataset(video_dir=video_dir, frame_list_file=training_list_file, transform=transform)
test_dataset = SingleVideoDataset(video_dir=video_dir, frame_list_file=testing_list_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

# Define the SiamTPN model
class SiamTPN(nn.Module):
    def __init__(self, d_model=464, num_heads=8, pooling_size=2, pooling_stride=2, padding=7):
        super(SiamTPN, self).__init__()
        self.backbone = Backbone()
        self.tpn = PABlock(d_model=d_model, num_heads=num_heads, pooling_size=pooling_size, pooling_stride=pooling_stride)
        self.depthwise_corr = DepthwiseCorrelation(padding=padding)
        self.classification_head = ClassificationHead(in_channels=d_model, num_classes=2)
        self.regression_head = RegressionHead(in_channels=d_model, num_coords=4)

    def forward(self, template, search):
        # Extract features from template and search images
        P3_template, P4_template, P5_template = self.backbone(template)
        P3_search, P4_search, P5_search = self.backbone(search)

        # Process features through the TPN
        template_features = self.tpn(P3_template, P4_template, P5_template)
        search_features = self.tpn(P3_search, P4_search, P5_search)

        # Perform depth-wise correlation
        correlation_maps = self.depthwise_corr(template_features, search_features)

        # Classification and regression
        classification_output = self.classification_head(correlation_maps)
        regression_output = self.regression_head(correlation_maps)

        return classification_output, regression_output

# Initialize the SiamTPN model
model = SiamTPN(num_heads=8, pooling_size=2, pooling_stride=2, padding=7).to(device)

# Define loss functions and optimizer
classification_criterion = nn.CrossEntropyLoss()
regression_criterion = nn.SmoothL1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
print("Training is starting")
for epoch in range(num_epochs):
    model.train()
    total_classification_loss = 0.0
    total_regression_loss = 0.0

    for batch_idx, (frames, annotations) in enumerate(train_loader):
        optimizer.zero_grad()

        template_frames, search_frames = frames
        template_annotations, search_annotations = annotations

        # Move data to the same device as the model
        template_frames = template_frames.to(device)
        search_frames = search_frames.to(device)
        template_annotations = torch.tensor(template_annotations).to(device)
        search_annotations = torch.tensor(search_annotations).to(device)

        # Forward pass
        classification_output, regression_output = model(template_frames, search_frames)

        # Compute classification and regression losses
        classification_labels = torch.ones(template_annotations.shape[0], dtype=torch.long, device=device)  # assuming object always present
        classification_loss = classification_criterion(classification_output, classification_labels)
        regression_loss = regression_criterion(regression_output, search_annotations)
        loss = classification_loss + regression_loss

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_classification_loss += classification_loss.item()
        total_regression_loss += regression_loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Classification Loss: {total_classification_loss:.4f}, Regression Loss: {total_regression_loss:.4f}")

print("Training is finished")

# Save the trained model
torch.save(model.state_dict(), 'siamtpn_model.pth')
