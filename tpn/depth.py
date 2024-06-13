import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthwiseCorrelation(nn.Module):
    def __init__(self, padding=0):
        super(DepthwiseCorrelation, self).__init__()
        self.padding = padding

    def forward(self, template_features, search_features):
        B, C, H_t, W_t = template_features.shape
        B_s, C_s, H_s, W_s = search_features.shape

        assert C == C_s, "The number of channels in template and search features must be the same."
        assert B == B_s, "The batch size of template and search features must be the same."

        correlation_maps = []

        # Perform depth-wise correlation using template features as kernels
        for i in range(C):
            template_feature = template_features[:, i:i+1, :, :]  # Shape: [B, 1, H_t, W_t]
            search_feature = search_features[:, i:i+1, :, :]  # Shape: [B, 1, H_s, W_s]

            batch_correlation_maps = []
            for b in range(B):
                # Perform convolution with padding
                correlation_map = F.conv2d(search_feature[b:b+1], template_feature[b:b+1], padding=self.padding)  
                batch_correlation_maps.append(correlation_map)

            # Concatenate along the batch dimension
            batch_correlation_maps = torch.cat(batch_correlation_maps, dim=0)  # Shape: [B, 1, H_out, W_out]
            correlation_maps.append(batch_correlation_maps)

        # Concatenate along the channel dimension
        correlation_maps = torch.cat(correlation_maps, dim=1)  # Shape: [B, C, H_out, W_out]

        return correlation_maps


"""
# Example input tensors
B, C, H_t, W_t = 4, 256, 14, 14  # Batch size, channels, height, width for template
H_s, W_s = 14, 14  # Height and width for search
template_features = torch.randn(B, C, H_t, W_t)
search_features = torch.randn(B, C, H_s, W_s)

# Initialize the model with padding and perform forward pass
padding = (H_t // 2, W_t // 2)  # Padding to maintain spatial dimensions
print(padding)
model = DepthwiseCorrelation(padding=padding)
output = model(template_features, search_features)

# Print the shapes of the outputs
print("Template features shape =", template_features.shape)
print("Search features shape =", search_features.shape)
print("Output shape =", output.shape)

"""