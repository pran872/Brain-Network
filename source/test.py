import torch
import torchvision
import torchvision.transforms as transforms
import torchattacks
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import classification_report
from tqdm import tqdm
import random
import numpy as np
import argparse
import json
try:
    from models import *
    from simple_cnn import set_config_defaults
except ModuleNotFoundError:
    from source.models import *
    from source.simple_cnn import set_config_defaults
import os
import sys

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_test_data(
    transform,
    batch_size,
    num_workers,
    debug=False,
):
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
   
    if debug:
        test_subset = Subset(test_set, list(range(0, 200)))
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=1)
        return test_loader

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader

def add_gaussian_noise(images, std=0.1):
    noise = torch.randn_like(images) * std
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)

def test_model(
    model,
    test_loader,
    device,
    attacker=False,
    gaussian_std=False
):
    model.eval()
    correct, total = 0, 0
    y_pred, y_true = [], []
    for images, labels in tqdm(test_loader, desc="Testing", disable=not sys.stdout.isatty()):
        images, labels = images.to(device), labels.to(device)

        if attacker:
            with torch.enable_grad():
                model.zero_grad(set_to_none=True)
                images.requires_grad = True
                adv_images = attacker(images, labels).detach()
                outputs = model(adv_images)
                del adv_images
        elif gaussian_std:
            with torch.no_grad():
                noisy_images = add_gaussian_noise(images, std=gaussian_std)
                outputs = model(noisy_images)
        else:
            with torch.no_grad():
                outputs = model(images)

        if isinstance(outputs, tuple): 
            # For flex network
            outputs, _ = outputs

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        y_pred.append(predicted.cpu().numpy())
        y_true.append(labels.cpu().numpy())

        if attacker:
            del images, labels, outputs, predicted
            torch.cuda.empty_cache()
    
    y_pred = np.concatenate(y_pred)
    y_true = np.concatenate(y_true)
    
    test_acc = 100 * correct / total
    cls_report = classification_report(y_true, y_pred, output_dict=False)

    return test_acc, cls_report

def parse_args():
    parser = argparse.ArgumentParser(description="Pass config")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=True, 
        help="Config file. If not provided, defaults will be used."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True
    )
    parser.add_argument(
        "-d", "--debug", 
        action="store_true",
        required=False,
        help="Run on debug mode"
    )

    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, TypeError) as e:
        config = {}
    
    return config, args.debug, args.model

def load_model(model_pt, config, device):
    if config["model_type"] == "resnet18":
        model = build_resnet18_for_cifar10().to(device)
    elif config["model_type"] == "zoom":
        model = ZoomVisionTransformer(
            device=device, 
            num_classes=10, 
            use_pos_embed=config["use_pos_embed"],
            add_dropout=config["add_dropout"],
            mlp_end=config["mlp_end"],
            add_cls_token=config["add_cls_token"],
            num_layers=config["num_layers"],
            trans_dropout_ratio=config["trans_dropout_ratio"],
            standard_scale=config["standard_scale"]
        ).to(device)
    
    checkpoint = torch.load(model_pt, map_location=device)
    model.load_state_dict(checkpoint)
    return model

def main():
    config, debug, model_pt = parse_args()
    config = set_config_defaults(config)
    
    assert config["attacker"] in [False, "FGSM", "PGD"], "Invalid attacker"
    assert (config["attacker"] and config["epsilon"]) or not config["attacker"], "Invalid"
    assert (config["gaussian_std"] and isinstance(config["gaussian_std"], float)) or not config["gaussian_std"], "Invalid"
    assert config["model_type"] in ["zoom", "resnet18"], "Testing only supports zoom and resnet18 models"

    if config["attacker"] and config["gaussian_std"]:
        print("Both attacker and gaussian std passed. Performing only attack - no gaussian noise")
        config["gaussian_std"] = False

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_pt, config, device)
    if config["attacker"] == "FGSM":
        config["attacker"] = torchattacks.FGSM(model, eps=config["epsilon"])
        config["batch_size"] = 32
    elif config["attacker"] == "PGD":
        config["attacker"] = torchattacks.PGD(model, eps=config["epsilon"], alpha=2/255, steps=10)
        config["batch_size"] = 32
    
    norm_means, norm_stds = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_means, norm_stds)])
    
    test_dataloader = load_test_data(transform, config["batch_size"], config["num_workers"], debug)
    acc, cls_report = test_model(
        model,
        test_dataloader,
        device,
        config["attacker"],
        config["gaussian_std"]
    )
    log_dir = os.path.dirname(model_pt)
    base_name = os.path.basename(model_pt).replace(".pt", "")
    if debug:
        results_fname = f"{log_dir}/test_manual_debug_{base_name}.txt"
    else:
        results_fname = f"{log_dir}/test_manual_{base_name}.txt"
    with open(results_fname, "w") as f:
        f.write(f"Test accuracy: {acc}")
        f.write(cls_report)
    print(f"Test accuracy: {acc}")

if __name__ == "__main__":
    main()