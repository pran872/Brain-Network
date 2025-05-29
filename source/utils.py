''' Helper functions '''
import torch
from torch.utils.data import DataLoader
from timm.data import Mixup
import numpy as np
import random
import logging

def get_logger(log_dir):
    print(log_dir)
    logger = logging.getLogger("debug_log")
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(f"{log_dir}/debug_log.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
    return logger

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

    return torch.Generator().manual_seed(seed)

def add_gaussian_noise(images, std=0.1):
    noise = torch.randn_like(images) * std
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)

def log_grad(writer, model, epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)

def process_gamma(gamma, gamma_by_class, labels):
    if gamma.ndim == 4:
        gamma = gamma.mean(dim=1).view(-1)
    for g, label in zip(gamma, labels):
        gamma_by_class[label.item()].append(g.item())
    return gamma_by_class

def get_mixup_fn(num_classes):
    return Mixup(
        mixup_alpha=0.8,  # controls mixing strength | 0 means no mixing, >1 means a lot of blending
        cutmix_alpha=1.0,  # CutMix strength - 1 encourage larger patches be cut
        cutmix_minmax=None, # all patch sizes valid
        prob=0.5,          # apply CutMix or MixUp 50% of the time
        switch_prob=0.5,   # probability to switch between CutMix and MixUp
        mode='batch',      # apply same CutMix/MixUp to all images in one batch 
        label_smoothing=0.1,
        num_classes=num_classes
    )

def compute_auxiliary_loss(auxiliary_loss, gamma, attn_maps):
    if "gamma_var_loss" in auxiliary_loss:
        max_allowed_var = 1
        lambda_param = auxiliary_loss["gamma_var_loss"]
        raw_var = torch.var(gamma, dim=0).mean()
        clipped_var = torch.clamp(raw_var, max=max_allowed_var)
        gamma_loss = lambda_param * -clipped_var
    elif "attention_entropy_loss" in auxiliary_loss:
        lambda_param = auxiliary_loss["attention_entropy_loss"]
        eps = 1e-8
        attn_maps = attn_maps.clamp(min=eps)
        entropy = -torch.sum(attn_maps * torch.log(attn_maps), dim=-1)
        gamma_loss = lambda_param * entropy.mean()
    else:
        gamma_loss = False
    return gamma_loss

def compute_mean_std(dataset):
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in loader:
        images = images.reshape(images.size(0), images.size(1), -1) # [B, 3, 224, 224] to [B, 3, 50176]
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += images.size(0)

    mean /= total_images
    std /= total_images
    return mean, std

class EarlyStopping:
    def __init__(self, patience=5, min_diff=0.0):
        self.patience = patience
        self.min_diff = min_diff
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_diff:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
