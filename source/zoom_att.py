import torch.nn as nn
from torchvision.models import resnet18

class ResNetBackbone(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity() # no maxpool

        self.stem = nn.Sequential(
            base.conv1, base.bn1, base.relu,
            base.layer1, base.layer2, base.layer3
        )
        self.out_dim = out_channels 

    def forward(self, x):
        feat_map = self.stem(x) # [B, 256, 8, 8]
        pooled = feat_map.mean(dim=[2, 3]) # [B, 256]
        return feat_map, pooled
    
class ZoomController(nn.Module):
    def __init__(self, in_dim, out_dim=1): 
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.Softplus() # +ve output only
        )

    def forward(self, x):
        return self.mlp(x) # [B, out_dim]

class ZoomAttention(nn.Module):
    def __init__(self, dim, num_heads, dist_matrix, dropout=0.0, standard_scale=False):
        super().__init__()
        self.num_heads = num_heads
        if standard_scale:
            self.scale = (dim // num_heads) ** -0.5
        else:
            self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.register_buffer('dist_matrix', dist_matrix) # [N, N]

    def forward(self, x, gamma): # gamma: [B, 1]
        B, N, D = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D_head]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        gamma = gamma.view(B, 1, 1, 1) # [B, 1, 1, 1]
        dist = self.dist_matrix.unsqueeze(0).unsqueeze(0) # [1, 1, N, N]
        attn_scores = attn_scores - gamma * dist

        attn_weights = self.attn_drop(attn_scores.softmax(dim=-1))
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, D)

        return self.proj_drop(self.proj(out))

class ZoomTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dist_matrix, mlp_ratio=4.0, dropout_ratio=0.0, standard_scale=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ZoomAttention(dim, num_heads, dist_matrix, dropout_ratio, standard_scale)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x, gamma):
        x = x + self.attn(self.norm1(x), gamma)
        x = x + self.mlp(self.norm2(x))
        return x