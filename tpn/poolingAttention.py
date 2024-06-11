import torch
import torch.nn as nn
import torch.nn.functional as F

class PoolingAttention(nn.Module):
    def __init__(self, d_model, num_heads, pooling_size):
        super(PoolingAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.pooling_size = pooling_size
        self.head_dim = d_model // num_heads

        assert self.head_dim * num_heads == d_model, "d_model must be divisible by num_heads"

        # Projection layer for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        
        # Output projection layer
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Normalization layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        
        # Linear projections after pooling
        self.k_proj = nn.Linear(self.head_dim, self.head_dim)
        self.v_proj = nn.Linear(self.head_dim, self.head_dim)

    def forward(self, x):
        B, N, C = x.shape

        # Step 1: Project input to Q, K, V
        qkv = self.qkv_proj(x)  # Shape: [B, N, 3 * d_model]
        
        # Step 2: Reshape and split Q, K, V
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # Shape: [B, N, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: [3, B, num_heads, N, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into Q, K, V; each of shape: [B, num_heads, N, head_dim]

        # Step 3: Reshape K and V for pooling
        H = W = int(N ** 0.5)  # Assuming N is a perfect square
        k = k.permute(0, 1, 3, 2).reshape(B * self.num_heads, self.head_dim, H, W)  # Shape: [B * num_heads, head_dim, H, W]
        v = v.permute(0, 1, 3, 2).reshape(B * self.num_heads, self.head_dim, H, W)  # Shape: [B * num_heads, head_dim, H, W]

        # Step 4: Pooling K and V
        k = F.avg_pool2d(k, self.pooling_size, stride=self.pooling_size)  # Shape reduced by pooling
        v = F.avg_pool2d(v, self.pooling_size, stride=self.pooling_size)  # Shape reduced by pooling

        # Step 5: Flatten pooled K and V
        k = k.reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # Shape: [B, num_heads, new_N, head_dim]
        v = v.reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # Shape: [B, num_heads, new_N, head_dim]

        # Step 6: Linear projection after pooling
        k = self.k_proj(k)  # Shape: [B, num_heads, new_N, head_dim]
        v = self.v_proj(v)  # Shape: [B, num_heads, new_N, head_dim]

        # Step 7: Compute attention
        attn_weights = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # Shape: [B, num_heads, N, new_N]
        attn_weights = attn_weights.softmax(dim=-1)  # Shape: [B, num_heads, N, new_N]
        attn_output = attn_weights @ v  # Shape: [B, num_heads, N, head_dim]

        # Step 8: Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)  # Shape: [B, N, C]

        # Step 9: Apply output projection
        attn_output = self.out_proj(attn_output)  # Shape: [B, N, C]
        
        # Step 10: Residual connection and normalization
        x = x + attn_output
        x = self.norm1(x)

        # Step 11: Apply MLP and final normalization
        x = x + self.mlp(x)
        x = self.norm2(x)

        return x
    

class PABlock(nn.Module):
    def __init__(self, d_model, num_heads, pooling_size):
        super(PABlock, self).__init__()
        self.pooling_attention = PoolingAttention(d_model, num_heads, pooling_size)

    def forward(self, P3, P4, P5):
        B, N4, C = P4.shape  # Use P4 as the reference size
        _, N3, _ = P3.shape
        _, N5, _ = P5.shape
        
        # Step 1: Apply pooling attention to each input matrix
        output_P3 = self.pooling_attention(P3)
        output_P4 = self.pooling_attention(P4)
        output_P5 = self.pooling_attention(P5)

        # Step 2: Resize outputs to match the size of P4
        output_P3_resized = F.interpolate(output_P3.permute(0, 2, 1).view(B, C, int(N3 ** 0.5), int(N3 ** 0.5)), size=(int(N4 ** 0.5), int(N4 ** 0.5)), mode='bilinear').view(B, C, N4).permute(0, 2, 1)
        output_P5_resized = F.interpolate(output_P5.permute(0, 2, 1).view(B, C, int(N5 ** 0.5), int(N5 ** 0.5)), size=(int(N4 ** 0.5), int(N4 ** 0.5)), mode='bilinear').view(B, C, N4).permute(0, 2, 1)

        # Step 3: Add the resized outputs together
        combined_output = output_P3_resized + output_P4 + output_P5_resized

        # Step 4: Apply pooling attention twice to the combined result
        combined_output = self.pooling_attention(combined_output)
        final_output = self.pooling_attention(combined_output)

        return final_output

# Example usage
B = 32  # Batch size
C = 256  # Feature dimension
num_heads = 8
pooling_size = 2

# Different sizes for P3, P4, P5
P3 = torch.randn(B, 49, C)  # Sequence length 49 (7x7)
P4 = torch.randn(B, 64, C)  # Sequence length 64 (8x8)
P5 = torch.randn(B, 81, C)  # Sequence length 81 (9x9)

multi_input_pa = PABlock(d_model=C, num_heads=num_heads, pooling_size=pooling_size)
output = multi_input_pa(P3, P4, P5)
print(output.shape)  # Expected output shape: [B, 64, C]