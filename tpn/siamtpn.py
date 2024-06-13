import torch
import torch.nn as nn
import numpy as np
from shuffleNet import Backbone
from head import ClassificationHead, RegressionHead
from poolingAttention import PABlock
from depth import DepthwiseCorrelation
from PIL import Image, ImageDraw
from torchvision import transforms
import logging
import math
import matplotlib.pyplot as plt

# Set up logging configuration
logging.basicConfig(level=logging.CRITICAL + 1, format='%(asctime)s - %(levelname)s - %(message)s')

# Uncomment to see the debuging messages
#logging.getLogger().setLevel(logging.DEBUG)



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

# Function to draw bounding box on an image
def draw_bounding_box(image_path, bbox):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)
    draw.rectangle(bbox, outline='red', width=3)
    image.show()

# Function to display a tensor as an image
def display_tensor_image(tensor_image):
    # Remove the batch dimension
    tensor_image = tensor_image.squeeze(0)
    # Denormalize the image
    inv_transform = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )
    tensor_image = inv_transform(tensor_image)
    # Convert the tensor to a numpy array
    np_image = tensor_image.permute(1, 2, 0).numpy()
    # Clip the values to be in the valid range [0, 1]
    np_image = np.clip(np_image, 0, 1)
    # Display the image
    plt.imshow(np_image)
    plt.axis('off')  # Turn off axis
    plt.show()

        
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
        print("Temprale features,",template_features.shape )
        search_features = self.tpn(P3_search, P4_search, P5_search)

    
        # Perform depth-wise correlation
        correlation_maps = self.depthwise_corr(template_features, search_features)

        # Classification and regression
        classification_output = self.classification_head(correlation_maps)
        regression_output = self.regression_head(correlation_maps)

        return classification_output, regression_output
    

    def predict_bbox(self, regression_output, classification_output):
        # Get the index of the maximum classification score
        classification_scores = torch.softmax(classification_output, dim=1)[0, 1]  # Get the positive class scores
        max_score, max_pos = torch.max(classification_scores.view(-1), dim=0)
        max_pos = torch.unravel_index(max_pos, classification_scores.shape)
        
        # Get the regression values at the position of the max classification score
        dx = regression_output[0, 0, max_pos[0], max_pos[1]]
        dy = regression_output[0, 1, max_pos[0], max_pos[1]]
        dw = regression_output[0, 2, max_pos[0], max_pos[1]]
        dh = regression_output[0, 3, max_pos[0], max_pos[1]]

        return [dx, dy, dw, dh]
    
def display_tensor_image_with_bbox(tensor_image, bbox):
    # Convert the tensor to a NumPy array and unnormalize it
    tensor_image = tensor_image.squeeze(0)
    tensor_image = tensor_image.permute(1, 2, 0).numpy()
    tensor_image = tensor_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    tensor_image = np.clip(tensor_image, 0, 1)
    
    plt.imshow(tensor_image)
    
    # Plot the bounding box
    x, y, w, h = bbox
    rect = plt.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)
    
    plt.axis('off')
    plt.show()


# Example usage
if __name__ == "__main__":
    # Path to the input image
    image_path = '../test/doggo.png'
    image = load_image(image_path)
    print("Image shape: ", image.shape)
    # Display the image
   # Display the tensor as an image
    display_tensor_image(image)

    # Initialize the Siamese tracker
    tracker = SiamTPN()

    # Make predictions using the same image as both template and search for testing
    classification_output, regression_output = tracker(image, image)

    print("The classification matrix is ", classification_output.shape)
    print("The regression matrix is ", regression_output.shape)

# Predict the bounding box
    pred_bbox = tracker.predict_bbox(regression_output, classification_output)
    print("Predicted bounding box: ", pred_bbox)

    # Convert the predicted bounding box to image coordinates
    x, y, w, h = pred_bbox
    x = x.item() * 224
    y = y.item() * 224
    w = w.item() * 224
    h = h.item() * 224
    pred_bbox = [x, y, w, h]

    # Display the image with the predicted bounding box
    display_tensor_image_with_bbox(image, pred_bbox)





