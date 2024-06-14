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

class LaSOTSubsetDataset(Dataset):
    def __init__(self, data_dir, selected_classes, video_list_file, transform=None):
        self.data_dir = data_dir
        self.selected_classes = selected_classes
        self.transform = transform
        self.video_list = self.load_video_list(video_list_file)
        self.template_search_pairs, self.annotations = self.load_data()

    def load_video_list(self, video_list_file):
        with open(video_list_file, 'r') as f:
            video_list = f.read().splitlines()
        return video_list

    def load_data(self):
        template_search_pairs = []
        annotations = []
        print(self.video_list)
        # Load data only for the selected classes
        for class_name in self.selected_classes:
            class_dir = os.path.join(self.data_dir, class_name)
            for video_name in self.video_list:
                item_path = os.path.join(class_dir, video_name)
                video_path = os.path.join(item_path, 'img')
                print("video path", video_path)
                frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')])
                groundtruth_file = os.path.join(item_path, 'groundtruth.txt')
                print("ground truth", groundtruth_file)
                # Read ground truth annotations
                with open(groundtruth_file, 'r') as f:
                    annots = f.readlines()
                
                print("frame images", len(frame_files))
                # Create template from the first frame and its bounding box
                template_frame = Image.open(frame_files[0]).convert('RGB')
                x, y, w, h = [int(x) for x in annots[0].strip().split(',')]
                template_crop = template_frame.crop((x, y, x + w, y + h))
                
                for i in range(1, len(frame_files)):
                    search_frame = Image.open(frame_files[i]).convert('RGB')
                    search_bbox = [float(x) for x in annots[i].strip().split(',')]
                    template_search_pairs.append((template_crop, search_frame))
                    annotations.append([template_bbox, search_bbox])  # Using the template's bbox and current frame's bbox
        return template_search_pairs, annotations
    
    def __len__(self):
        return len(self.template_search_pairs)
    
    def __getitem__(self, idx):
        template_frame, search_frame = self.template_search_pairs[idx]
        template_annotation, search_annotation = self.annotations[idx]
        if self.transform:
            template_frame = self.transform(template_frame)
            search_frame = self.transform(search_frame)
        return (template_frame, search_frame), (template_annotation, search_annotation)

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the selected classes
selected_classes = ['airplane', 'basketball', 'bear', 'bicycle', 'bird', 'boat', 'book', 'car', 'cat', 'chameleon', 'crab', 'crocodile', 'deer', 'dog', 'fox', 'gecko', 'goldfish', 'hippo', 'monkey', 'person', 'pig', 'shark', 'turtle']  # Replace with actual class names

# Path to the video list files
training_list_file = '../../dataset/training_set.txt'
testing_list_file = '../../dataset/testing_set.txt'

# Load the LaSOT subset dataset
data_dir = '../../dataset'
train_dataset = LaSOTSubsetDataset(data_dir=data_dir, selected_classes=selected_classes, video_list_file=training_list_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

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
        print("Template features,", template_features.shape)
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
print("training is starting")
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
