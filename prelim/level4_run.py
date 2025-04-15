from custom_models import Flex2D, CustomViTHybrid

# !pip install torchvision timm tqdm

import torch
import sys
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import os, random, json
import pandas as pd
from tqdm import tqdm
import matplotlib.cm as cm
import torchvision.transforms.functional as TF
from torch.nn import MaxPool2d, functional as F
from torch import cat
import time

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Activation Maps

def plot_activation_grid(tensor, name, save_dir, max_channels=16):
    act = tensor[0][:max_channels]  # [C, H, W]
    grid = vutils.make_grid(act.unsqueeze(1), nrow=4, normalize=True, scale_each=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.title(f"Activations: {name}")
    plt.savefig(os.path.join(save_dir, f"activation_{name}.png"))
    plt.close()

def overlay_activation_on_image(input_img, activation_map, channel=0, save_path=None, alpha=0.5, title=""):
    act = activation_map[channel].unsqueeze(0).unsqueeze(0)
    act_resized = F.interpolate(act, size=input_img.shape[1:], mode="bilinear", align_corners=False)[0, 0]
    act_resized = act_resized.cpu().numpy()
    act_resized -= act_resized.min()
    act_resized /= act_resized.max() + 1e-8
    heatmap = cm.jet(act_resized)[..., :3]
    heatmap = torch.tensor(heatmap).permute(2, 0, 1)
    img = input_img.cpu() * 0.5 + 0.5
    overlay = (1 - alpha) * img + alpha * heatmap
    plt.figure(figsize=(4, 4))
    plt.imshow(overlay.permute(1, 2, 0).clamp(0, 1))
    plt.axis("off")
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def train_model(model, config):
    start_time = time.time()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    train_loss, val_loss, train_acc, val_acc = [], [], [], []
    activation_maps = {}

    def save_activation(name):
        def hook(model, input, output):
            activation_maps[name] = output.detach().cpu()
        return hook

    if config["log_activations"]:
        first_conv = next((m for m in model.conv_blocks if isinstance(m, (nn.Conv2d, Flex2D))), None)
        if first_conv:
            first_conv.register_forward_hook(save_activation("conv1"))

        if config["use_flex"]:
            for i, m in enumerate(model.conv_blocks):
                if isinstance(m, Flex2D):
                    m.register_forward_hook(save_activation(f"flex_{i}"))

    for epoch in range(config["epochs"]):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{config['epochs']}", leave=False, disable=True)

        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if config["log_gradients"]:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} grad mean: {param.grad.abs().mean().item():.4f}")
            optimizer.step()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()

        scheduler.step()
        train_loss.append(running_loss / len(trainloader))
        train_acc.append(100. * correct / total)

        # Validation
        model.eval()
        loss_val, correct_val, total_val = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()
                loss_val += loss.item()
        val_loss.append(loss_val / len(valloader))
        val_acc.append(100. * correct_val / total_val)

        print(f"Epoch {epoch+1}: Train Acc={train_acc[-1]:.2f}%, Val Acc={val_acc[-1]:.2f}%")

    # Test
    model.eval()
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()
    test_accuracy = 100. * test_correct / test_total
    print(f"\nFinal Test Accuracy: {test_accuracy:.2f}%")

    total_time = time.time() - start_time
    print(f"\n Total training time: {total_time:.2f} seconds")

    return train_loss, val_loss, train_acc, val_acc, test_accuracy, activation_maps, total_time

if __name__ == "__main__":
    experiment_config = {
        "use_flex": True,
        "cnn_channels": 16,
        "embed_dim": 64,
        "depth": 2,
        "heads": 4,
        "dropout": 0.1,
        "lr": 0.001,
        "weight_decay": 1e-4,
        "epochs": 2,
        "cnn_depth": 4,
        "flex_positions": [2],
        "use_batch_norm": False,
        "log_activations": True,
        "log_gradients": False,
    }
    pbs_jobid = os.environ.get("PBS_JOBID", "localtest")
    experiment_config["save_dir"] = f"experiment_logs/exp_{pbs_jobid}"
    os.makedirs(experiment_config["save_dir"], exist_ok=True)
    
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(">>> Debug: top-level code running")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    train_size = int(0.8 * len(full_trainset))
    val_size = len(full_trainset) - train_size
    trainset, valset = random_split(full_trainset, [train_size, val_size])
    trainloader = DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    valloader = DataLoader(valset, batch_size=256, shuffle=False, num_workers=2)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)


    model = CustomViTHybrid(
        cnn_channels=experiment_config["cnn_channels"],
        embed_dim=experiment_config["embed_dim"],
        depth=experiment_config["depth"],
        heads=experiment_config["heads"],
        use_flex=experiment_config["use_flex"],
        cnn_depth=experiment_config["cnn_depth"],
        flex_positions=experiment_config["flex_positions"],
        use_batch_norm=experiment_config["use_batch_norm"],
        device=device
    )

    train_loss, val_loss, train_acc, val_acc, test_acc, activations, total_time = train_model(model, experiment_config)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.suptitle(f"Test Accuracy: {test_acc:.2f}%", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{experiment_config['save_dir']}/curves.png")
    plt.show()

    df = pd.DataFrame({
        "epoch": list(range(1, experiment_config["epochs"] + 1)),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": [test_acc] * len(train_loss),
        "total_training_testing_time": [total_time] * len(train_loss)
    })

    df.to_csv(f"{experiment_config['save_dir']}/log.csv", index=False)

    with open(f"{experiment_config['save_dir']}/config.json", "w") as f:
        json.dump(experiment_config, f, indent=4)

    if experiment_config["log_activations"]:
        for name, act in activations.items():
            plot_activation_grid(act, name, experiment_config["save_dir"])

        input_img, _ = trainset[0]
        for name, act in activations.items():
            overlay_activation_on_image(
                input_img=input_img,
                activation_map=act[0],
                channel=0,
                save_path=os.path.join(experiment_config["save_dir"], f"overlay_{name}_ch0.png"),
                title=f"{name} Channel 0"
            )

    print("All training complete and logs saved.")