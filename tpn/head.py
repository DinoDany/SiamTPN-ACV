import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

# Set up logging configuration
logging.basicConfig(level=logging.CRITICAL + 1, format='%(asctime)s - %(levelname)s - %(message)s')

# Uncomment to see the debuging messages
#logging.getLogger().setLevel(logging.DEBUG)

# Define the classes
class ClassificationHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(ClassificationHead, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.conv_layers(x)

class RegressionHead(nn.Module):
    def __init__(self, in_channels, num_coords):
        super(RegressionHead, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, num_coords, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
    def forward(self, x):
        return self.conv_layers(x)



# Test cases
def test_classification_head():
    B, C, H, W = 4, 256, 14, 14  # Batch size, channels, height, width
    num_classes = 2
    x = torch.randn(B, C, H, W)
    print("Testing the classification function")
    print(" Inputs x shape =", x.shape)
    
    model = ClassificationHead(in_channels=C, num_classes=num_classes)
    output = model(x)
    
    
    assert output.shape == (B, num_classes, H, W), f"Expected shape {(B, num_classes, H, W)}, but got {output.shape}"
    print(" Output x shape =", output.shape)
    print("ClassificationHead test passed.")

def test_regression_head():
    B, C, H, W = 4, 256, 14, 14  # Batch size, channels, height, width
    num_coords = 4
    x = torch.randn(B, C, H, W)
    print("Testing the regression function")
    print(" Inputs x shape =", x.shape)

    model = RegressionHead(in_channels=C, num_coords=num_coords)
    output = model(x)
    
    assert output.shape == (B, num_coords, H, W), f"Expected shape {(B, num_coords, H, W)}, but got {output.shape}"
    print(" Output x shape =", output.shape)
    print("RegressionHead test passed.")


# Run the tests
test_classification_head()
test_regression_head()
""""""
