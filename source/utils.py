''' General utils that you could use across models and datasets - not integral to the pipeline'''
import torch
from torch.utils.data import DataLoader

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