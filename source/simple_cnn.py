'''Simple CNN'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchinfo import summary
from tqdm import tqdm
import time
import random
import numpy as np
import argparse
import json
from models import *
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import logging
import sys


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_logger(log_dir):
    logger = logging.getLogger("debug_log")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{log_dir}/debug_log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
    return logger

def load_data(transform, train_split, batch_size, num_workers):
    full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    train_size = int(train_split * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader

def log_grad(writer, model, epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)

def train_model(model, epochs, train_loader, val_loader, device, optimizer, criterion, writer, logger, verbose=True):

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} [Train]", disable=not sys.stdout.isatty()):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            if writer and batch_idx == 0: 
                log_grad(writer, model, epoch)
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Validation 
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        end = time.time()

        if writer:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

        if verbose:
            logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}% | "
              f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}% | Time: {end - start:.2f}s")

    return model, train_losses, val_losses, train_accuracies, val_accuracies

def test_model(model, test_loader, device, criterion, writer, logger, verbose=True):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing", disable=not sys.stdout.isatty()):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total_loss += criterion(outputs, labels)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_acc = 100 * correct / total
    test_loss = (total_loss/len(test_loader)).item()
    if writer:
        writer.add_text("Test/Accuracy", f"{test_acc:.2f}%")
        writer.add_text("Test/Loss", f"{test_loss:.4f}")

    if verbose:
        logger.info(f"Test Accuracy: {test_acc:.2f}% | Test Loss: {test_loss:.2f}")
    return test_acc, test_loss

def parse_args():
    parser = argparse.ArgumentParser(description="A simple argparse example")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=False, 
        default=None,
        help="Config file. If not provided, defaults will be used."
    )
    parser.add_argument(
        "-v", "--verbose", 
        action="store_true",
        required=False,
        help="Increase output verbosity"
    )

    args = parser.parse_args()
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except (json.JSONDecodeError, TypeError) as e:
        if args.verbose:
            # print(e)
            print("Config not provided. Using defualts")
        config = {}
    
    return config, args.verbose

def main():
    try:
        config, verbose = parse_args()
        seed = config.get("seed", 42)
        train_split = config.get('train_split', 0.9)
        batch_size = config.get("batch_size", 256)
        num_workers = config.get("num_workers", 1)
        model_type = config.get("model_type", "fast_cnn")
        optimizer = config.get("optimizer", "adam")
        criterion = config.get("criterion", "CE")
        lr = config.get("lr", 0.001)
        epochs = config.get("epochs", 1)
        writer = config.get("writer", True)
        run_name = config.get('run_name', 'run')

        assert model_type in ["fast_cnn"], "Invalid model type"
        assert optimizer in ["adam"], "Invalid optimizer"
        assert criterion in ["CE"], "Invalid criterion"

        time_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = f"runs/{run_name}_{time_stamp}"
        os.makedirs(log_dir, exist_ok=True)

        if writer:
            writer = SummaryWriter(log_dir=log_dir)
            writer.add_text("config", json.dumps(config, indent=2))

        set_seed(seed)
        logger = get_logger(log_dir)
        logger.info("Job Started")
        logger.info("Config:\n" + json.dumps(config, indent=2))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if verbose:
            logger.info(f"Using device: {device}")
            logger.info(f"Training for {epochs} epochs")
            logger.info(f"Logging run: {log_dir}")
            if writer:
                logger.info(f"Tensorboard running on port: {6006}")

        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        train_loader, val_loader, test_loader = load_data(transform, train_split, batch_size, num_workers)

        if model_type == "fast_cnn":
            model = FastCNN().to(device)
            summary_str = summary(model, input_size=(1, 3, 32, 32), device=device, verbose=0)
            with open(f"{log_dir}/model_summary.txt", "w") as f:
                f.write(str(summary_str))
        
        if optimizer == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
        if criterion == "CE":
            criterion = nn.CrossEntropyLoss()

        model, train_losses, val_losses, train_acc, val_acc = train_model(
            model, 
            epochs, 
            train_loader, 
            val_loader,
            device,
            optimizer,
            criterion,
            writer,
            logger,
            verbose
        )
        torch.save(model.state_dict(), f"{log_dir}/{run_name}_{time_stamp}_{epochs}_model.pt")
        test_acc, test_loss = test_model(model, test_loader, device, criterion, writer, logger, verbose)

        with open(f"{log_dir}/metrics_{run_name}_{time_stamp}.csv", "w+") as f:
            f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
            f.write(f"Test acc and loss:,{test_acc},{test_loss},,\n")
            for i in range(epochs):
                f.write(f"{i},{train_losses[i]},{val_losses[i]},{train_acc[i]},{val_acc[i]}\n")

        if writer:
            dummy_input = torch.randn(1, 3, 32, 32).to(device)
            writer.add_graph(model, dummy_input)
            writer.close()
        logger.info("\nJob Completed")

    except Exception as e:
        logger.exception("Error occured")
        raise  # Optional: re-raise to trigger error code exit
    

if __name__ == "__main__":
    main()
