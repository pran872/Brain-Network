import torch.nn as nn
import torch.nn.functional as F
import torch
from torchvision.models import resnet18
import torchvision.transforms as T

class ZoomVisionTransformer(nn.Module):
    def __init__(
        self,
        device,
        num_classes,
        use_pos_embed=False,
        add_dropout=False,
        mlp_end=False,
        add_cls_token=False,
        num_layers=2,
        trans_dropout_ratio=0.0,
        standard_scale=False,
        embed_dim=256,
        num_heads=4,
        resnet_layers=3,
        multiscale_tokenisation=False,
        freeze_resnet_early=False,
        gamma_per_head=False,
        use_token_mixer=False,
        remove_zoom=False,
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
        
        self.register_buffer("dist_matrix", self.compute_token_distance_matrix(device=device))
        self.backbone = ResNetBackbone(resnet_layers=resnet_layers, freeze_early=freeze_resnet_early)
        self.token_proj = nn.Linear(self.backbone.out_dim, embed_dim)
        self.zoom_controller = ZoomController(self.backbone.out_dim, out_dim=1, num_heads=num_heads, gamma_per_head=gamma_per_head)
        
        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.use_pos_embed:
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
                self.dist_matrix, 
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
            tokens = block(tokens, gamma)

        out = tokens[:, 0] if self.add_cls_token else tokens.mean(dim=1)
        out = self.mlp_head(out) if self.mlp_end else self.cls_head(out)

        if return_gamma:
            return out, gamma
        else:
            return out

    def compute_token_distance_matrix(self, h=8, w=8, device="cpu"):
        coords = torch.stack(torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij'
        ), dim=-1).view(-1, 2).float()

        if self.multiscale_tokenisation:
            coords_4x4 = torch.stack(torch.meshgrid(
                torch.linspace(0, h-1, 4, device=device),
                torch.linspace(0, w-1, 4, device=device),
                indexing='ij'
            ), dim=-1).view(-1, 2).float()
            coords = torch.cat([coords, coords_4x4], dim=0)

            coords_1x1 = torch.tensor([[h / 2, w / 2]], device=device)
            coords = torch.cat([coords, coords_1x1], dim=0) 

        dist = torch.cdist(coords, coords, p=2)

        if self.add_cls_token: # Expand to fit the extra token
            # Create on same device!!
            cls_row = torch.zeros(1, dist.size(1), device=dist.device)
            cls_col = torch.zeros(dist.size(0) + 1, 1, device=dist.device)
            dist = torch.cat([cls_row, dist], dim=0)
            dist = torch.cat([cls_col, dist], dim=1)

        return dist


class BrainiT(nn.Module):
    def __init__(
        self,
        device,
        use_retinal_layer=True,
        num_classes=10,
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
    ):
        super().__init__()
        self.use_retinal_layer = use_retinal_layer
        self.use_pos_embed = use_pos_embed
        self.add_dropout = add_dropout
        self.mlp_end = mlp_end
        self.add_cls_token = add_cls_token
        self.multiscale_tokenisation = multiscale_tokenisation

        if self.add_dropout or self.add_cls_token:
            # Dropout and cls token used with pos embeds
            self.use_pos_embed = True
        
        self.register_buffer("dist_matrix", self.compute_token_distance_matrix(device=device))
        if self.use_retinal_layer:
            self.retinal_sampling_layer = LearnableFoveation()
        self.backbone = ResNetBackbone(resnet_layers=resnet_layers, freeze_early=freeze_resnet_early)
        self.token_proj = nn.Linear(self.backbone.out_dim, embed_dim)
        self.zoom_controller = ZoomController(self.backbone.out_dim, out_dim=1, num_heads=num_heads, gamma_per_head=gamma_per_head)
        
        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        if self.use_pos_embed:
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
                self.dist_matrix, 
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

    def forward(self, x, return_cx_cy=False):
        if self.use_retinal_layer:
            x = self.retinal_sampling_layer(x, return_cx_cy)
            if isinstance(x, tuple):
                x, cx, cy = x
        feat_map, pooled = self.backbone(x) # [B, 512, 8, 8], [B, 512]
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
            tokens = block(tokens, gamma)

        out = tokens[:, 0] if self.add_cls_token else tokens.mean(dim=1)
        out = self.mlp_head(out) if self.mlp_end else self.cls_head(out)

        if return_cx_cy:
            return out, cx, cy
        else:
            return out

    def compute_token_distance_matrix(self, h=8, w=8, device="cpu"):
        coords = torch.stack(torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij'
        ), dim=-1).view(-1, 2).float()

        if self.multiscale_tokenisation:
            coords_4x4 = torch.stack(torch.meshgrid(
                torch.linspace(0, h-1, 4, device=device),
                torch.linspace(0, w-1, 4, device=device),
                indexing='ij'
            ), dim=-1).view(-1, 2).float()
            coords = torch.cat([coords, coords_4x4], dim=0)

            coords_1x1 = torch.tensor([[h / 2, w / 2]], device=device)
            coords = torch.cat([coords, coords_1x1], dim=0) 

        dist = torch.cdist(coords, coords, p=2)

        if self.add_cls_token: # Expand to fit the extra token
            # Create on same device!!
            cls_row = torch.zeros(1, dist.size(1), device=dist.device)
            cls_col = torch.zeros(dist.size(0) + 1, 1, device=dist.device)
            dist = torch.cat([cls_row, dist], dim=0)
            dist = torch.cat([cls_col, dist], dim=1)

        return dist

class ResNetBackbone(nn.Module):
    def __init__(self, out_channels=512, resnet_layers=4, freeze_early=False):
        super().__init__()
        base = resnet18(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity() # no maxpool

        if resnet_layers==4:
            self.upsample_features = True
            self.stem = nn.Sequential(
                base.conv1, base.bn1, base.relu,
                base.layer1, base.layer2, base.layer3, base.layer4
            )
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

        if freeze_early: # Freezes base.conv1, base.bn1, base.relu, base.layer1
            for param in list(self.stem[:4].parameters()):
                param.requires_grad = False

    def forward(self, x):
        feat_map = self.stem(x) # [B, 512, 4, 4]
        if self.upsample_features:
            feat_map = self.upsample(feat_map) # [B, 512, 8, 8]
        pooled = feat_map.mean(dim=[2, 3]) # [B, 512]
        return feat_map, pooled
    
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
        dist_matrix,
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

        self.register_buffer('dist_matrix', dist_matrix) # [N, N]

    def forward(self, x, gamma): # gamma: [B, 1]
        B, N, D = x.shape
        H = self.num_heads
        qkv = self.qkv(x).reshape(B, N, 3, H, D // H).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B, H, N, D_head]

        attn_scores = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        dist = self.dist_matrix.unsqueeze(0).unsqueeze(0) # [1, 1, N, N]
        if self.gamma_per_head:
            gamma = gamma.view(B, H, 1, 1)
        else:
            gamma = gamma.view(B, 1, 1, 1) # [B, 1, 1, 1]

        if not self.remove_zoom:
            attn_scores = attn_scores - gamma * dist
        
        attn_weights = self.attn_drop(attn_scores.softmax(dim=-1))
        out = (attn_weights @ v).transpose(1, 2).contiguous().view(B, N, D)

        return self.proj_drop(self.proj(out))

class ZoomTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        dist_matrix,
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
        self.attn = ZoomAttention(dim, num_heads, dist_matrix, gamma_per_head, dropout_ratio, standard_scale, remove_zoom)
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

    def forward(self, x, gamma):
        x = x + self.attn(self.norm1(x), gamma)

        if self.use_token_mixer:
            x_normed = self.token_norm(x)
            x_mixed = self.token_mixer(x_normed.transpose(1, 2))
            x = x + x_mixed.transpose(1, 2)  

        x = x + self.mlp(self.norm2(x))
        return x
    

class LearnableFoveation(nn.Module):
    """Training it to find the centroid"""

    def __init__(self, fovea_radius=10, alpha=1, input_size=32):
        super().__init__()
        self.fovea_radius = fovea_radius
        self.alpha = alpha
        self.input_size = input_size
        self.blur = T.GaussianBlur(kernel_size=3, sigma=0.8)

        self.pay_attn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=1),
            # nn.Softmax(dim=-1)
        )

    def forward(self, x, return_cx_cy=False):
        B, C, H, W = x.shape
        attn = self.pay_attn(x)
        attn_flat = attn.view(B, -1)
        attn_soft = F.softmax(attn_flat, dim=1)
        attn = attn_soft.view(B, 1, H, W)

        coords_y = torch.linspace(0, H - 1, H, device=x.device) # centroid x
        coords_x = torch.linspace(0, W - 1, W, device=x.device) # centroid y
        grid_y, grid_x = torch.meshgrid(coords_y, coords_x, indexing="ij")
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)

        cx = torch.sum(attn * grid_x, dim=(2, 3))
        cy = torch.sum(attn * grid_y, dim=(2, 3))
            
        blurred = self.blur(x.cpu()).to(x.device)

        foveated_imgs = []
        for i in range(B):
            
            # Make dist mask and use to blur the outside of the fovea
            xx, yy = torch.meshgrid(
                torch.arange(W, device=x.device),
                torch.arange(H, device=x.device),
                indexing="xy"
            )
            dist = ((xx - cx[i])**2 + (yy - cy[i])**2).float().sqrt()
            mask = (dist <= self.fovea_radius).float().unsqueeze(0) # [1, H, W]
            mask = mask.expand(C, -1, -1) # [C, H, W]

            sharp = x[i]     
            blurred_img = blurred[i] 

            foveated = self.alpha * (mask * sharp) + (1 - mask) * blurred_img
            foveated_imgs.append(foveated)

        #     to_pil = T.ToPILImage()
        #     original_img = to_pil(x[i]) 
        #     blurred_img = to_pil(blurred_img) 
        #     foveated_img = to_pil(foveated)
        #     original_img.save("og_image.png") 
        #     blurred_img.save("blurred_img.png") 
        #     foveated_img.save("foveated_img.png") 
        #     print('breaking')
        #     break
        # print('exiting')
        # exit()
        if return_cx_cy:
            return torch.stack(foveated_imgs), cx.mean(dim=0), cy.mean(dim=0)
        else:
            return torch.stack(foveated_imgs)
    
class FixedFoveation:
    def __init__(self, fovea_size=16, alpha=0.7):
        self.fovea_size = fovea_size
        self.alpha = alpha
        self.blur = T.GaussianBlur(kernel_size=5, sigma=2)

    def __call__(self, img):
        C, H, W = img.shape
        fovea = img[:, 
                    H//2 - self.fovea_size//2 : H//2 + self.fovea_size//2,
                    W//2 - self.fovea_size//2 : W//2 + self.fovea_size//2]
        
        high_res = F.interpolate(fovea.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False).squeeze(0)
        low_res = self.blur(img)
        
        # Blend 
        foveated_img = self.alpha * high_res + (1 - self.alpha) * low_res

        # Save sample image
        # to_pil = T.ToPILImage()
        # original_img = to_pil(img) 
        # cropped_img = to_pil(fovea) 
        # high_res_img = to_pil(high_res)
        # low_res_img = to_pil(low_res)
        # blended_img = to_pil(foveated_img)
        # original_img.save("og_image.png") 
        # cropped_img.save("cropped_img.png") 
        # high_res_img.save("high_res_img.png") 
        # low_res_img.save("low_res_img.png") 
        # blended_img.save("blended_img.png") 
        # exit()

        return foveated_img