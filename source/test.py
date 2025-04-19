import torch
from tqdm import tqdm
from models import *
import random
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import glob
import os
import sys

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_resnet18_for_cifar10():
    model = resnet18(weights=None)
    
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity() 
    model.fc = nn.Linear(512, 10)
    return model

def test_model(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    y_pred, y_true = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", disable=not sys.stdout.isatty()):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            if isinstance(outputs, tuple): 
                # For flex network
                outputs, _ = outputs

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            y_pred.append(predicted.cpu().numpy())
            y_true.append(labels.cpu().numpy())
    
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    
    test_acc = 100 * correct / total
    print("Test/Accuracy", f"{test_acc:.2f}%")

    cls_report = classification_report(y_true, y_pred, output_dict=False)

    return test_acc, cls_report

    
if __name__ == "__main__":
    log_dirs = [
        "/Users/pranathipoojary/Imperial/FYP/Brain-Network/runs/zoom_20250418-202010_6_layers",
        "/Users/pranathipoojary/Imperial/FYP/Brain-Network/runs/zoom_20250418-202224_4_layers",
        ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [
        ZoomVisionTransformer(device=device, num_classes=10, num_layers=6).to(device),
        ZoomVisionTransformer(device=device, num_classes=10, num_layers=4).to(device)
        ]
    for log_dir, model in zip(log_dirs, models):
        
        model_pt = glob.glob(os.path.join(log_dir, "**", "*.pt"), recursive=True)
        assert len(model_pt)==1, f"Please ensure there is only one model path in the directory not {len(model_pt)}"
        model_pt = model_pt[0]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(42)
        checkpoint = torch.load(model_pt, map_location=device)
        model.load_state_dict(checkpoint)

        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
        test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=1)
        test_acc, cls_report = test_model(model, test_loader, device)

        with open(f"{log_dir}/cls_report_last_manual.txt", "w+") as f:
            f.write(cls_report)
        print("DONE")