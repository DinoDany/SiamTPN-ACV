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
                template_bbox = [float(x) for x in annots[0].strip().split(',')]
                template_crop = self.crop_image(template_frame, template_bbox)
                
                for i in range(1, len(frame_files)):
                    search_frame = Image.open(frame_files[i]).convert('RGB')
                    search_bbox = [float(x) for x in annots[i].strip().split(',')]
                    template_search_pairs.append((template_crop, search_frame))
                    annotations.append([template_bbox, search_bbox])  # Using the template's bbox and current frame's bbox
        return template_search_pairs, annotations
    
    def crop_image(self, image, bbox):
        x, y, w, h = bbox
        return image.crop((x, y, x + w, y + h))
    
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
selected_classes = ['basketball']
# Path to the video list files
training_list_file = '../../dataset/training_set.txt'
testing_list_file = '../../dataset/testing_set.txt'

# Load the LaSOT subset dataset
data_dir = '../../dataset'
train_dataset = LaSOTSubsetDataset(data_dir=data_dir, selected_classes=selected_classes, video_list_file=training_list_file, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
