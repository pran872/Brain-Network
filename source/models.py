import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
try:
    from flex import *
    from zoom_att import *
    from brainit import LearnableFoveation
except ModuleNotFoundError:
    from source.flex import *
    from source.zoom_att import *
    from brainit import LearnableFoveation

def build_resnet(dataset_type, pretrained=False):
    weights = "IMAGENET1K_V1" if pretrained else None

    if dataset_type == "cifar10":
        model = resnet18(weights=weights)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity() 
        model.fc = nn.Linear(512, 10)
        return model
    
    elif dataset_type == "stanford_dogs":
        model = resnet18(weights=weights)
        model.fc = nn.Linear(512, 120)
        return model

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
            x, conv_ratio = self.flex(x)
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
        if self.use_flex:
            return self.mlp_head(x), conv_ratio
        else:
            return self.mlp_head(x)

class ZoomVisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes,
        num_patches=None,
        use_pos_embed=True,
        add_dropout=False,
        mlp_end=False,
        add_cls_token=True,
        num_layers=2,
        trans_dropout_ratio=0.0,
        standard_scale=True,
        embed_dim=256,
        num_heads=4,
        resnet_layers=4,
        multiscale_tokenisation=False,
        freeze_resnet_early=False,
        gamma_per_head=False,
        use_token_mixer=False,
        remove_zoom=False,
        pretrained=False,
    ):
        super().__init__()
        self.use_pos_embed = use_pos_embed
        self.add_dropout = add_dropout
        self.mlp_end = mlp_end
        self.add_cls_token = add_cls_token
        self.multiscale_tokenisation = multiscale_tokenisation

        if self.add_dropout or self.add_cls_token:
            # Dropout and cls token used with pos embeds
            self.use_pos_embed = True
        
        self.dist_matrix = None
        self.backbone = ResNetBackbone(resnet_layers=resnet_layers, freeze_early=freeze_resnet_early, pretrained=pretrained)
        
        self.token_proj = nn.Linear(self.backbone.out_dim, embed_dim)
        self.zoom_controller = ZoomController(self.backbone.out_dim, out_dim=1, num_heads=num_heads, gamma_per_head=gamma_per_head)
        
        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.use_pos_embed:
            if num_patches is None:
                num_patches = 81 if self.multiscale_tokenisation else 64
            pos_embed_patches = num_patches + 1 if self.add_cls_token else num_patches
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02) # std by convention
            if self.add_dropout:
                self.dropout = nn.Dropout(0.1)
            
        self.transformer_blocks = nn.ModuleList([
            ZoomTransformerBlock(
                embed_dim, 
                num_heads,
                gamma_per_head=gamma_per_head,
                dropout_ratio=trans_dropout_ratio, 
                standard_scale=standard_scale,
                use_token_mixer=use_token_mixer,
                remove_zoom=remove_zoom
            )
            for _ in range(num_layers)
        ])

        if self.mlp_end:
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, 128),
                nn.GELU(),
                nn.Linear(128, num_classes)
            )
        else:
            self.cls_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, num_classes)
            )

    def forward(self, x, return_gamma=False):
        feat_map, pooled = self.backbone(x) # [B, 512, 8, 8], [B, 512]

        if self.dist_matrix is None:
            H, W = feat_map.shape[-2:]
            self.dist_matrix = self._compute_token_distance_matrix(h=H, w=W, device=feat_map.device)

        gamma = self.zoom_controller(pooled) # [B, 1]
        tokens = feat_map.flatten(2).transpose(1, 2) # [B, 64, 512]
        if self.multiscale_tokenisation:
            pooled_4x4 = F.adaptive_avg_pool2d(feat_map, output_size=(4, 4))
            tokens_4x4 = pooled_4x4.flatten(2).transpose(1, 2)
            pooled_1x1 = F.adaptive_avg_pool2d(feat_map, 1).reshape(x.shape[0], self.backbone.out_dim)
            tokens_1x1 = pooled_1x1.unsqueeze(1)
            tokens = torch.cat([tokens, tokens_4x4, tokens_1x1], dim=1)
        tokens = self.token_proj(tokens) # [B, N, D]

        if self.add_cls_token:
            cls_token = self.cls_token.expand(tokens.shape[0], 1, -1)
            tokens = torch.cat((cls_token, tokens), dim=1)

        if self.use_pos_embed:
            tokens = tokens + self.pos_embed
            if self.add_dropout:
                tokens = self.dropout(tokens)

        for block in self.transformer_blocks:
            tokens = block(tokens, gamma, self.dist_matrix)

        out = tokens[:, 0] if self.add_cls_token else tokens.mean(dim=1)
        out = self.mlp_head(out) if self.mlp_end else self.cls_head(out)

        if return_gamma:
            return out, gamma
        else:
            return out

    def _compute_token_distance_matrix(self, h=8, w=8, device="cpu"):
        coords = torch.stack(torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij'
        ), dim=-1).reshape(-1, 2).float()

        if self.multiscale_tokenisation:
            coords_4x4 = torch.stack(torch.meshgrid(
                torch.linspace(0, h-1, 4, device=device),
                torch.linspace(0, w-1, 4, device=device),
                indexing='ij'
            ), dim=-1).view(-1, 2).float()
            coords_1x1 = torch.tensor([[h / 2, w / 2]], device=device)
            coords = torch.cat([coords, coords_4x4, coords_1x1], dim=0) 

        dist = torch.cdist(coords, coords, p=2) # 2-norm euclidean

        if self.add_cls_token: # Expand to fit the extra token
            # Create on same device!!
            cls_row = torch.zeros(1, dist.size(1), device=dist.device)
            cls_col = torch.zeros(dist.size(0) + 1, 1, device=dist.device)
            dist = torch.cat([cls_row, dist], dim=0)
            dist = torch.cat([cls_col, dist], dim=1)

        return dist

class BrainiT(ZoomVisionTransformer):
    def __init__(
        self,
        num_classes,
        embed_dim,
        num_patches=None,
        use_retinal_layer=True,

    ):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_patches=num_patches,
            resnet_layers=4
        )
        self.use_retinal_layer = use_retinal_layer
        if self.use_retinal_layer:
            self.retinal_sampling_layer = LearnableFoveation()
    
    def forward(self, x, return_cx_cy=False, return_gamma=False):
        if self.use_retinal_layer:
            x = self.retinal_sampling_layer(x, return_cx_cy=return_cx_cy)
            if isinstance(x, tuple):
                x, cx, cy = x
        out = super().forward(x, return_gamma=return_gamma)
        if isinstance(out, tuple):
            out, gamma = out

        return_vars = [out]

        if return_cx_cy: 
            return_vars.extend([cx, cy]) 
        if return_gamma:
            return_vars.append(gamma)

        return return_vars
     
class ZoomVisionTransformer224(ZoomVisionTransformer):
    def __init__(self, num_classes, embed_dim, pretrained):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_patches=196,
            resnet_layers=4,
        )

        self.backbone = ResNetBackbone224(resnet_layers=4, freeze_early=False, pretrained=pretrained)

class BrainiT224(BrainiT):
    def __init__(self, num_classes, embed_dim, pretrained, use_retinal_layer=True):
        super().__init__(
            num_classes=num_classes,
            embed_dim=embed_dim,
            num_patches=196,
            use_retinal_layer=use_retinal_layer
        )

        self.backbone = ResNetBackbone224(resnet_layers=4, freeze_early=False, pretrained=pretrained)