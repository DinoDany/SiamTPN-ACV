import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import math

# Set up logging configuration
logging.basicConfig(level=logging.CRITICAL + 1, format='%(asctime)s - %(levelname)s - %(message)s')

# Uncomment to see the debuging messages
#logging.getLogger().setLevel(logging.DEBUG)

class PoolingAttention_layer1(nn.Module):
    def __init__(self, d_model, num_heads, pooling_size, pooling_stride):
        super(PoolingAttention_layer1, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
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

    def forward(self, x, q4):
        B, C, N = x.shape
        N4 = q4.shape[2]

        logging.debug("N4 %s", N4)

        # Step 1: Permute to [B, N, C] for projection
        x = x.permute(0, 2, 1)  # Shape: [B, N, C]

        logging.debug("X %s", x.shape)

        # Step 2: Project input to K, V (Q is from q4)
        qkv = self.qkv_proj(x)  # Shape: [B, N, 3 * d_model]
        logging.debug("qkv %s", qkv.shape)

        
        # Step 3: Reshape and split Q, K, V
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # Shape: [B, N, 3, num_heads, head_dim]
        logging.debug("after reshape qkv %s", qkv.shape)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: [3, B, num_heads, N, head_dim]
        logging.debug("after permute qkv %s", qkv.shape)


        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into Q, K, V; each of shape: [B, num_heads, N, head_dim]
        logging.debug("Q:  %s", q.shape)
        logging.debug("K:  %s", k.shape)
        logging.debug("V:  %s", v.shape)


        # Step 4: Reshape K and V for pooling
        H = W = int(N ** 0.5)  # Assuming N is a perfect square
        k = k.permute(0, 1, 3, 2).reshape(B * self.num_heads, self.head_dim, H, W)  # Shape: [B * num_heads, head_dim, H, W]
        v = v.permute(0, 1, 3, 2).reshape(B * self.num_heads, self.head_dim, H, W)  # Shape: [B * num_heads, head_dim, H, W]
        logging.debug("K after permute and reshape:  %s", k.shape)
        logging.debug("V after permute and reshape:  %s", v.shape)

        # Step 5: Pooling K and V
        k = F.avg_pool2d(k, self.pooling_size, stride=self.pooling_stride)  # Shape reduced by pooling
        v = F.avg_pool2d(v, self.pooling_size, stride=self.pooling_stride)  # Shape reduced by pooling
        logging.debug("K after pooling:  %s", k.shape)
        logging.debug("V after pooling:  %s", v.shape)



        # Step 6: Flatten pooled K and V
        k = k.reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # Shape: [B, num_heads, new_N, head_dim]
        v = v.reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # Shape: [B, num_heads, new_N, head_dim]
        logging.debug("K flatten pooled:  %s", k.shape)
        logging.debug("V flatten pooled:  %s", v.shape)


        # Step 7: Linear projection after pooling
        k = self.k_proj(k)  # Shape: [B, num_heads, new_N, head_dim]
        v = self.v_proj(v)  # Shape: [B, num_heads, new_N, head_dim]
        logging.debug("K after linear projection:  %s", k.shape)
        logging.debug("V after linear projection:  %s", v.shape)

        # Step 8: Compute attention
        attn_weights = (q4 @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # Shape: [B, num_heads, N, new_N]
        logging.debug("Q4 @ K:  %s", attn_weights.shape)
   
        attn_weights = attn_weights.softmax(dim=-1)  # Shape: [B, num_heads, N, new_N]
        attn_output = attn_weights @ v  # Shape: [B, num_heads, N, head_dim]
        logging.debug("attn_output  %s", attn_output.shape)


        # Step 9: Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N4, C)  # Shape: [B, N, C]
        logging.debug("After concatenating the heads  %s", attn_output.shape)


        # Step 10: Apply output projection
        attn_output = self.out_proj(attn_output)  # Shape: [B, N, C]
        logging.debug("attention output final projection %s", attn_output.shape)


        # Step 11: Residual connection and normalization
        x = x[:, :N4, :]  # Match the sequence length to N4
        logging.debug("Matching x to q4%s", x.shape)

        x = x + attn_output  # Shape: [B, N, C]
        x = self.norm1(x)  # Shape: [B, N, C]
        logging.debug("X after add and norm %s", x.shape)
        
        # Step 12: Apply MLP and final normalization
        x = x + self.mlp(x)
        x = self.norm2(x)  # Shape: [B, N, C]
        logging.debug("X after MLP %s", x.shape)


        # Step 13: Permute back to [B, C, N]
        x = x.permute(0, 2, 1)  # Shape: [B, C, N]
        logging.debug("X after last permutation %s", x.shape)
        return x

class PoolingAttention_layer2(nn.Module):
    def __init__(self, d_model, num_heads, pooling_size, pooling_stride):
        super(PoolingAttention_layer2, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.pooling_size = pooling_size
        self.head_dim = d_model // num_heads
        self.pooling_stride =pooling_stride

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
        B, C, N = x.shape

        # Step 1: Permute to [B, N, C] for projection
        x = x.permute(0, 2, 1)  # Shape: [B, N, C]

        # Step 2: Project input to Q, K, V
        qkv = self.qkv_proj(x)  # Shape: [B, N, 3 * d_model]
        
        # Step 3: Reshape and split Q, K, V
        qkv = qkv.reshape(B, N, 3, self.num_heads, self.head_dim)  # Shape: [B, N, 3, num_heads, head_dim]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # Shape: [3, B, num_heads, N, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]  # Split into Q, K, V; each of shape: [B, num_heads, N, head_dim]

        # Step 4: Reshape K and V for pooling
        H = W = int(N ** 0.5)  # Assuming N is a perfect square
        k = k.permute(0, 1, 3, 2).reshape(B * self.num_heads, self.head_dim, H, W)  # Shape: [B * num_heads, head_dim, H, W]
        v = v.permute(0, 1, 3, 2).reshape(B * self.num_heads, self.head_dim, H, W)  # Shape: [B * num_heads, head_dim, H, W]

        # Step 5: Pooling K and V
        k = F.avg_pool2d(k, self.pooling_size, stride=self.pooling_stride)  # Shape reduced by pooling
        v = F.avg_pool2d(v, self.pooling_size, stride=self.pooling_stride)  # Shape reduced by pooling

        # Step 6: Flatten pooled K and V
        k = k.reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # Shape: [B, num_heads, new_N, head_dim]
        v = v.reshape(B, self.num_heads, self.head_dim, -1).permute(0, 1, 3, 2)  # Shape: [B, num_heads, new_N, head_dim]

        # Step 7: Linear projection after pooling
        k = self.k_proj(k)  # Shape: [B, num_heads, new_N, head_dim]
        v = self.v_proj(v)  # Shape: [B, num_heads, new_N, head_dim]

        # Step 8: Compute attention
        attn_weights = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)  # Shape: [B, num_heads, N, new_N]
        attn_weights = attn_weights.softmax(dim=-1)  # Shape: [B, num_heads, N, new_N]
        attn_output = attn_weights @ v  # Shape: [B, num_heads, N, head_dim]

        # Step 9: Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)  # Shape: [B, N, C]

        # Step 10: Apply output projection
        attn_output = self.out_proj(attn_output)  # Shape: [B, N, C]

        # Step 11: Residual connection and normalization
        x = x + attn_output  # Shape: [B, N, C]
        x = self.norm1(x)  # Shape: [B, N, C]

        # Step 12: Apply MLP and final normalization
        x = x + self.mlp(x)
        x = self.norm2(x)  # Shape: [B, N, C]

        # Step 13: Permute back to [B, C, N]
        x = x.permute(0, 2, 1)  # Shape: [B, C, N]

        return q, x

class PABlock(nn.Module):
    def __init__(self, d_model, num_heads, pooling_size, pooling_stride):
        super(PABlock, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.pooling_stride = pooling_stride

        self.pooling_attention_layer1 = PoolingAttention_layer1(d_model, num_heads, pooling_size, 4)
        self.pooling_attention_layer2 = PoolingAttention_layer2(d_model, num_heads, pooling_size, pooling_stride)
        
        # Projection layer for Q, K, V
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)

    def forward(self, P3, P4, P5):

        q_4, output_P4 = self.pooling_attention_layer2(P4)
        print("Output_P4 ", output_P4.shape)


        # Step 3: Apply pooling attention to each input matrix
        output_P3 = self.pooling_attention_layer1(P3, q_4)
        #print("Output_P3 ", output_P3.shape)
        output_P5 = self.pooling_attention_layer1(P5, q_4)
        #print("Output_P5 ", output_P5.shape)

        # Step 4: Add the resized outputs together
        combined_output = output_P3 + output_P4 + output_P5
        #print("addition of Ps ", combined_output.shape)

        # Step 5: Apply pooling attention twice to the combined result
        _, combined_output = self.pooling_attention_layer2(combined_output)
        #print("Pooling attention after combination", combined_output.shape)
        _, final_output = self.pooling_attention_layer2(combined_output)

        #print("Final Output", final_output.shape)

        #Last reshape

        B, C, N = final_output.shape
        H = W = int(math.sqrt(N))  # Calculate H and W assuming it's a square

        reshaped_output = final_output.view(B, C, H, W)
        

        return reshaped_output
"""
# Example usage
B = 1  # Batch size
C = 464  # Feature dimension
num_heads = 8
pooling_size = 2
pooling_stride = 2

# Different sizes for P3, P4, P5
P3 = torch.randn(B, C, 196)  # Sequence length 196 (14x14)
P4 = torch.randn(B, C, 49)  # Sequence length 49 (7x7)
P5 = torch.randn(B, C, 49)  # Sequence length 49 (7x7)

pa_block= PABlock(d_model=C, num_heads=num_heads, pooling_size=pooling_size,pooling_stride = pooling_stride)
output = pa_block(P3, P4, P5)

print(output.shape)



#Testing layer by layer

test = PoolingAttention_layer2(d_model=C, num_heads=num_heads, pooling_size=pooling_size)
test2 = PoolingAttention_layer1(d_model=C, num_heads=num_heads, pooling_size=pooling_size)

q, output_2 = test(P4)
print("Q shape:", q.shape)
output_1 = test2(P3, q)
output_3 = test2(P5, q)

print(output_1.shape)
print(output_2.shape)
print(output_3.shape)

combined = output_1 + output_2 + output_3
print(combined.shape)

_, output_4 = test(combined)
print(output_4.shape)

"""