import torch.nn as nn
from torchvision.models import resnet18
import torch.nn.functional as F

class ResNetBackbone(nn.Module):
    def __init__(
        self, 
        out_channels=512, 
        resnet_layers=4, 
        freeze_early=False, 
        upsample_features=True,
        downsample_features=False,
        pretrained=False
    ):
        super().__init__()
        self.upsample_features = upsample_features
        self.downsample_features = downsample_features
        weights = "IMAGENET1K_V1" if pretrained else None
        
        base = resnet18(weights=weights)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity() # no maxpool

        if resnet_layers==4:
            self.stem = nn.Sequential(
                base.conv1, base.bn1, base.relu,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
            if self.upsample_features:
                self.upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(512, 512, kernel_size=1)
                )
        else:
            self.upsample_features = False
            self.stem = nn.Sequential(
                base.conv1, base.bn1, base.relu,
                base.layer1, base.layer2, base.layer3
            )
            out_channels = 256
        self.out_dim = out_channels 

        if freeze_early: # Freezes base.conv1, base.bn1, base.relu, base.layer1 for sample efficiency stuff
            for param in list(self.stem[:4].parameters()):
                param.requires_grad = False

    def forward(self, x):
        feat_map = self.stem(x) # [B, 512, 4, 4]
        if self.upsample_features:
            feat_map = self.upsample(feat_map) # [B, 512, 8, 8]
        elif self.downsample_features:
            feat_map = F.adaptive_avg_pool2d(feat_map, output_size=(14, 14)) # 196 tokens
        pooled = feat_map.mean(dim=[2, 3]) # [B, 512]
        return feat_map, pooled

class ResNetBackbone224(ResNetBackbone):
    def __init__(self, out_channels=512, resnet_layers=4, freeze_early=False, pretrained=False):
        super().__init__(
            out_channels=out_channels, 
            resnet_layers=resnet_layers, 
            freeze_early=freeze_early,
            upsample_features=False,
            downsample_features=True,
            pretrained=pretrained
        )
    
    
class ZoomController(nn.Module):
    def __init__(self, in_dim, out_dim=1, num_heads=4, gamma_per_head=False): 
        super().__init__()
        self.gamma_per_head = gamma_per_head
        self.num_heads = num_heads

        if self.gamma_per_head:
            out_dim = self.num_heads

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.Softplus() # +ve output only
        )

    def forward(self, x):
        gamma = self.mlp(x) # [B, out_dim]
        if self.gamma_per_head:
            gamma = gamma.view(x.size(0), self.num_heads, 1, 1)
        return gamma

class ZoomAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        gamma_per_head,
        dropout=0.0,
        standard_scale=False,
        remove_zoom=False
    ):
        super().__init__()
        self.remove_zoom = remove_zoom
        self.num_heads = num_heads
        self.gamma_per_head = gamma_per_head
        if standard_scale:
            self.scale = (dim // num_heads) ** -0.5
        else:
            self.scale = dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # self.register_buffer('dist_matrix', dist_matrix) # [N, N]

    def forward(self, x, gamma, dist_matrix, return_attn_map=False): # gamma: [B, 1]
        B, N, D = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D_head]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        dist = dist_matrix.unsqueeze(0).unsqueeze(0) # [1, 1, N, N]
        if self.gamma_per_head:
            gamma = gamma.view(B, H, 1, 1)
        else:
            gamma = gamma.view(B, 1, 1, 1) # [B, 1, 1, 1]

        if not self.remove_zoom:
            attn_scores = attn_scores - gamma * dist
        
        attn_weights = self.attn_drop(attn_scores.softmax(dim=-1))
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, D)

        if return_attn_map:
            return self.proj_drop(self.proj(out)), attn_weights
        else:
            return self.proj_drop(self.proj(out))

class ZoomTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        gamma_per_head,
        mlp_ratio=4.0,
        dropout_ratio=0.0,
        standard_scale=False,
        use_token_mixer=False,
        remove_zoom=False
    ):
        super().__init__()
        self.use_token_mixer = use_token_mixer
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ZoomAttention(dim, num_heads, gamma_per_head, dropout_ratio, standard_scale, remove_zoom)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
        if self.use_token_mixer:
            self.token_norm = nn.LayerNorm(dim)
            self.token_mixer = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),  # Depthwise Conv
                nn.GELU()
            )

    def forward(self, x, gamma, dist_matrix, return_attn_map=False):
        attn_out = self.attn(self.norm1(x), gamma, dist_matrix, return_attn_map)
        if isinstance(attn_out, tuple):
            attn_out, attn_map = attn_out
        x = x + attn_out

        if self.use_token_mixer:
            x_normed = self.token_norm(x)
            x_mixed = self.token_mixer(x_normed.transpose(1, 2))
            x = x + x_mixed.transpose(1, 2)  

        x = x + self.mlp(self.norm2(x))
        if return_attn_map:
            return x, attn_map
        else:
            return x