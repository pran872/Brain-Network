import torch.nn as nn
import torch.nn.functional as F
from flex import *

class FastCNN(nn.Module):
    def __init__(self):
        super(FastCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class FastCNN2(nn.Module):
    def __init__(self):
        super(FastCNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class FlexNet(nn.Module):
    def __init__(self, device, image_dimensions=(32, 32)):
        super(FlexNet, self).__init__()

        self.image_dimensions = image_dimensions
        self.in_dimensions = (3, self.image_dimensions[0], self.image_dimensions[1])

        dimension_tracer = DimensionTracer(self.in_dimensions)
        class Conv2d_flex(Flex2D):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                self.out_dimensions = dimension_tracer.calculate_dimension()

                self.init_dimension_dependent_modules()
        
        class Conv2d(nn.Conv2d):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.in_dimensions = dimension_tracer.calculate_dimension()
                dimension_tracer(**kwargs)
                self.out_dimensions = dimension_tracer.calculate_dimension()

        self.conv_ratio = 0

        self.conv1 = Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = Conv2d_flex(device=device, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x, self.conv_ratio = self.conv3(x)
        x = self.pool(F.relu(self.bn3(x)))

        x = torch.flatten(x, start_dim=1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x, self.conv_ratio

class ConvViTHybrid(nn.Module):
    def __init__(self, device, patch_size=4, in_channels=3,
                 cnn_channels=32, embed_dim=64, depth=2, heads=4,
                 num_classes=10, use_flex=False):
        super().__init__()

        self.use_flex = use_flex
        self.device = device
        self.patch_size = patch_size

        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        if self.use_flex:
            self.flex = Flex2D(16, cnn_channels, kernel_size=3, stride=1, padding=1, device=self.device)
        else:
            self.conv2 = nn.Conv2d(16, cnn_channels, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)

        self.patch_embed = nn.Conv2d(cnn_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (8 // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.dropout = nn.Dropout(0.1)

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        B = x.size(0)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        if self.use_flex:
            x = self.flex(x)
        else:
            x = F.relu(self.conv2(x))
        x = self.pool2(x)

        x = self.patch_embed(x) # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, embed_dim]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.norm(x)
        x = self.transformer(x)
        x = self.dropout(x)

        x = x[:, 0]
        return self.mlp_head(x)
