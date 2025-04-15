import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
from torch.nn import MaxPool2d, functional as F
from torch import cat

class Flex2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, device, stride=1, padding=0, config=None):
        super().__init__()
        """
        # in dimensions: in this case [C, H, W]
        # --------
        # logits_mechanism: THRESHOLD or SPATIAL_ATTENTION_(1-3)
        # masking_mechanism: "SIGMOID", "STE", "SIGMOID_SMOOTHED", "STE_SIGMOID"
        # num_spatial_attention_block: int
        # logits_use_batchnorm: bool

        # about parameter vs variable:
        variable is almost deprecated and works the same as just plain tensor. And a Parameters is a specific Tensor that is marked as being a parameter from an nn.Module and so will be returned when calling .parameters() on this Module.
        """
        # -------- set configs --------
        # assert config, "Missing config file for Flex2D"
        # self.config = config
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.device = device

        # -------- Initialize layers --------
        self.flex_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.flex_pool = MaxPool2d(kernel_size, stride, padding)
        self.bn_logits = nn.BatchNorm2d(self.out_channels)

        # -------- Initialize monitored variables --------
        self.homogeneity = 0  # for monitoring the binariness of the mask later on
        self.conv_ratio = 0  # for later updating
        self.cp_identity_matrix = None  # store the matrix indicating the channel pool identity

    def channel_interpolate(self, tensor, out_channels):
        """
        Interpolates a tensor along the channel axis. This is for addressing the mismatch in the number of channels between the output of the maxpool layer and the output of the convolutional layer.
        """
        tensor = tensor.permute(0, 2, 1, 3)
        tensor = F.interpolate(tensor, [out_channels, tensor.size(3)], mode="bilinear")
        tensor = tensor.permute(0, 2, 1, 3)
        return tensor

    def channel_wise_maxpool(self, tensor_1, tensor_2):
        """
        Take two tensors of identical shape and return a tensor of the same shape using element-wise max pooling.
        Also returns the ratio of values from tensor_2 to the total.
        """
        assert tensor_1.shape == tensor_2.shape, "tensor_1 and tensor_2 must have the same shape"

        joint = cat([tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)], dim=-1)
        pooled, indices = torch.max(joint, dim=-1)

        # count values are from conv (tensor 2)
        with torch.no_grad():
            conv_ratio = (indices == 1).sum().item() / tensor_1.numel()
            cp_identity_matrix = (indices == 1).int()

        return pooled, conv_ratio, cp_identity_matrix

    def forward(self, x):
        """
        threshold can only be initialized when the output dimensions are known
        """
        # -------- make the raw conv and pool --------
        t_flex_pool = self.flex_pool(x)
        t_flex_conv = self.flex_conv(x)
        t_flex_pool = self.channel_interpolate(t_flex_pool, self.out_channels)

        # -------- get the binary mask --------
        output, self.conv_ratio, self.cp_identity_matrix = self.channel_wise_maxpool(t_flex_pool, t_flex_conv)
        # print("Conv ratio:", self.conv_ratio)
        return output

class CustomViTHybrid(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 cnn_channels=16, embed_dim=64, depth=2, heads=4,
                 num_classes=10, cnn_depth=4, flex_positions=None,
                 use_flex=True, relu_after_flex=True, use_batch_norm=False,
                 device="cpu"):
        super().__init__()

        self.use_flex = use_flex
        self.relu_after_flex = relu_after_flex
        self.device = device

        self.conv_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        input_channels = in_channels
        out_channels = cnn_channels

        for i in range(cnn_depth):
            # Use Flex2D or standard Conv2D
            if self.use_flex and flex_positions and i in flex_positions:
                layer = Flex2D(input_channels, out_channels, kernel_size=3, stride=1, padding=1, device=device)
            else:
                layer = nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1)

            self.conv_blocks.append(layer)

            if use_batch_norm:
                self.conv_blocks.append(nn.BatchNorm2d(out_channels))

            self.conv_blocks.append(nn.ReLU(inplace=True))

            if (i + 1) % 2 == 0:
                self.conv_blocks.append(self.pool)

            input_channels = out_channels
            if out_channels < 512:
                out_channels *= 2

        # Patch embedding for ViT
        self.patch_embed = nn.Conv2d(input_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // 4 // patch_size) ** 2

        # CLS token & position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # ViT encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # Classifier
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        for layer in self.conv_blocks:
            x = layer(x)

        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.transformer(x)
        x = x[:, 0]
        return self.mlp_head(x)
    