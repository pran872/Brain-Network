import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

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


