import torch
from tqdm import tqdm
from models import *
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
import sys

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", disable=not sys.stdout.isatty()):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple): 
                # For flex network
                outputs, conv_ratio = outputs

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100 * correct / total
    print("Test/Accuracy", f"{test_acc:.2f}%")

    return test_acc

    
if __name__ == "__main__":
    model_pt = "/Users/pranathipoojary/Imperial/FYP/Brain-Network/runs/zoom_20250418-155458/zoom_20250418-155458_e22_best_model.pt"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    
    model = ZoomVisionTransformer(device=device, num_classes=10).to(device)
    checkpoint = torch.load(model_pt, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.load_state_dict(checkpoint)

    transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=1)
    test_acc = test_model(model, test_loader, device)