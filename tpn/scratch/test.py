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


def load_video_list(video_list_file):
    with open(video_list_file, 'r') as f:
        video_list = f.read().splitlines()
    return video_list

def crop_image(image, bbox):
    x, y, w, h = bbox
    return image.crop((x, y, x + w, y + h))


# Path to the video list files
training_list_file = '../../dataset/training_set.txt'
testing_list_file = '../../dataset/testing_set.txt'

print(testing_list_file)

# Define the selected classes
selected_classes = ['airplane', 'basketball', 'bear', 'bicycle', 'bird', 'boat', 'book', 'car', 'cat', 'chameleon', 'crab', 'crocodile', 'deer', 'dog', 'fox', 'gecko', 'goldfish', 'hippo', 'monkey', 'person', 'pig', 'shark', 'turtle']  # Replace with actual class names

template_search_pairs = []
annotations = []
print(training_list_file)
data_dir = '../../dataset'
video_list = load_video_list(training_list_file)

for class_name in selected_classes:
    #print(class_name)
    class_dir = os.path.join(data_dir, class_name)
    #print(class_dir)
    for video_name in video_list:
                item_path = os.path.join(class_dir, video_name)
                video_path = os.path.join(item_path, 'img')
                


video_path = "../../dataset/basketball/basketball-16/img"
item_path = "../../dataset/basketball/basketball-16"
print("item_path", item_path)
print("video path", video_path)

frame_files = sorted([os.path.join(video_path, f) for f in os.listdir(video_path) if f.endswith('.jpg')])
print("Frame fiels type", type(frame_files))
print("Frame fiels len", len(frame_files))
print("Frame file 1", frame_files[0])

groundtruth_file = os.path.join(item_path, 'groundtruth.txt')
with open(groundtruth_file, 'r') as f:
    annots = f.readlines()
print(annots[0])

template_frame = Image.open(frame_files[0]).convert('RGB')

#template_frame.show()
template_bbox = [int(x) for x in annots[0].strip().split(',')]
print(template_bbox[0])
template_crop = crop_image(template_frame, template_bbox)
template_crop.show()

for i in range(1, len(frame_files)):
        search_frame = Image.open(frame_files[i]).convert('RGB')
        search_bbox = [float(x) for x in annots[i].strip().split(',')]
        template_search_pairs.append((template_crop, search_frame))
        annotations.append([template_bbox, search_bbox])  # Using the template's bbox and current frame's bbox

print("annotations: ", len(annotations))
print("pairs: ", len(template_search_pairs))