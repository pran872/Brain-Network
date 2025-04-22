import torch
import torch.nn.functional as F
import torchvision.transforms as T

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


