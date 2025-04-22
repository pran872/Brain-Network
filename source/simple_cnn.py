'''Simple CNN'''
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchattacks
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision.transforms.autoaugment import AutoAugmentPolicy
from torch_ema import ExponentialMovingAverage
from torchinfo import summary
from contextlib import nullcontext
from sklearn.metrics import classification_report
from tqdm import tqdm
import time
import random
import numpy as np
import argparse
import json
try:
    from models import *
    from brainit import *
except ModuleNotFoundError:
    from source.models import *
    from source.brainit import *

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import logging
import sys
import socket

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
    torch.use_deterministic_algorithms(True)

    return torch.Generator().manual_seed(seed)

def get_log_dir(run_name: str, time_stamp, base_env_var="OUTPUT_DIR") -> str:
    full_run_name = f"{run_name}_{time_stamp}"

    output_root = os.environ.get(base_env_var)

    if not output_root:
        hostname = socket.gethostname()
        if "login" in hostname or "node" in hostname or "rds" in os.getcwd():
            output_root = "/rds/general/user/psp20/home/Brain-Network/runs"
        else:
            output_root = "runs/round_2"

    log_dir = os.path.join(output_root, full_run_name)
    os.makedirs(log_dir, exist_ok=True)

    return log_dir

def get_logger(log_dir):
    logger = logging.getLogger("debug_log")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{log_dir}/debug_log.log")
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    fh.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)
    return logger

def class_balanced_subset(dataset, fraction):
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    indices = []

    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        selected = np.random.choice(cls_idx, size=int(len(cls_idx) * fraction), replace=False)
        indices.extend(selected)

    np.random.shuffle(indices) 
    return Subset(dataset, indices)

def few_shot_subset(dataset, n_per_class):
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    indices = []

    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        selected = np.random.choice(cls_idx, size=n_per_class, replace=False)
        indices.extend(selected)

    np.random.shuffle(indices)
    return Subset(dataset, indices)

def load_data(
    transform_train,
    transform_test,
    train_split,
    batch_size,
    num_workers,
    seed_worker_fn,
    seed_generator,
    downsample_fraction=0,
    few_shot=False,
    debug=False,
    logger=False,
    test_batch_size=False,
):
    if not test_batch_size:
        test_batch_size = batch_size

    full_train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
   
    if debug:
        subset_indices = list(range(0, 200))
        train_subset = Subset(full_train_set, subset_indices)
        val_subset = Subset(full_train_set, subset_indices)
        test_subset = Subset(test_set, subset_indices)
        
        train_loader = DataLoader(train_subset, batch_size=64, shuffle=True, num_workers=1, worker_init_fn=seed_worker_fn, generator=seed_generator)
        val_loader = DataLoader(val_subset, batch_size=64, shuffle=False, num_workers=1, worker_init_fn=seed_worker_fn, generator=seed_generator)
        test_loader = DataLoader(test_subset, batch_size=64, shuffle=False, num_workers=1, worker_init_fn=seed_worker_fn, generator=seed_generator)
        return train_loader, val_loader, test_loader

    if downsample_fraction > 0:
        full_train_set = class_balanced_subset(full_train_set, downsample_fraction)
    elif few_shot:
        full_train_set = few_shot_subset(full_train_set, few_shot)

    logger.info(f"Full train/val size with downsampling ({downsample_fraction}): {len(full_train_set)}")

    train_size = int(train_split * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker_fn, generator=seed_generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker_fn, generator=seed_generator)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker_fn, generator=seed_generator)

    return train_loader, val_loader, test_loader

def log_grad(writer, model, epoch):
    for name, param in model.named_parameters():
        if param.grad is not None:
            writer.add_histogram(f"gradients/{name}", param.grad, epoch)

def train_model(
    model,
    epochs,
    train_loader,
    val_loader,
    device,
    optimizer,
    criterion,
    scheduler,
    early_stopping,
    ema,
    writer,
    logger
):

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_model = {"val_loss": float("inf"),
                  "model_state": None,
                  "epoch": 0}
    gamma_by_class = {i: [] for i in range(10)}

    for epoch in range(epochs):
        start = time.time()
        model.train()
        total_loss, correct, total = 0, 0, 0

        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} [Train]", disable=not sys.stdout.isatty()):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            if isinstance(outputs, tuple): 
                # For flex network / zoomvit
                if len(outputs) == 2:
                    outputs, _ = outputs
    
            loss = criterion(outputs, labels)
            loss.backward()
            if writer and batch_idx == 0: 
                log_grad(writer, model, epoch)
            optimizer.step()
            if ema:
                ema.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Validation 
        if ema:
            context = ema.average_parameters()
        else:
            context = nullcontext()
        
        with context:
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    
                    if isinstance(model, ZoomVisionTransformer):
                        outputs, gamma = model(images, return_gamma=True)
                        if gamma.ndim == 4:
                            gamma = gamma.mean(dim=1).view(-1)
                        for g, label in zip(gamma, labels):
                            gamma_by_class[label.item()].append(g.item())
                    elif isinstance(model, BrainiT):
                        outputs, cx, cy = model(images, return_cx_cy=True)
                    elif isinstance(model, FlexNet):
                        outputs, conv_ratio = outputs
                    else:
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

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
    
        end = time.time()

        if writer:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LR", current_lr, epoch)

            if isinstance(model, FlexNet):
                writer.add_scalar("Conv_ratio", conv_ratio, epoch)
            elif isinstance(model, ZoomVisionTransformer):
                writer.add_scalar("Gamma/mean", gamma.mean().item(), epoch)
                writer.add_scalar("Gamma/std", gamma.std().item(), epoch)
                for c in range(10):
                    writer.add_scalar(f"Gamma/Class_{c}", np.mean(gamma_by_class[c]), epoch)
            elif isinstance(model, BrainiT):
                writer.add_scalar("Centroid_means/cx", cx, epoch)
                writer.add_scalar("Centroid_means/cy", cy, epoch)

        logger.info(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}% | "
            f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}% | Time: {end - start:.2f}s")
        
        if avg_val_loss < best_model["val_loss"]:
            best_model["val_loss"] = avg_val_loss
            best_model["model_state"] = model.state_dict()
            best_model["epoch"] = epoch
            logger.info(f"New best model saved (val loss = {best_model['val_loss']:.4f})")

        if early_stopping:
            early_stopping.step(avg_val_loss)
            if early_stopping.should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

    last_model = {
        "val_loss": avg_val_loss,
        "model_state": model.state_dict(),
        "epoch": epoch
        }

    return best_model, last_model, train_losses, val_losses, train_accuracies, val_accuracies

def add_gaussian_noise(images, std=0.1):
    noise = torch.randn_like(images) * std
    noisy_images = images + noise
    return torch.clamp(noisy_images, 0.0, 1.0)

def test_model(model,
    test_loader,
    device,
    writer,
    logger,
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
            outputs = outputs[0]

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
    if writer:
        writer.add_text("Test/Accuracy", f"{test_acc:.2f}%")

    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    cls_report = classification_report(y_true, y_pred, output_dict=False)

    return test_acc, cls_report

def parse_args():
    parser = argparse.ArgumentParser(description="Pass a config file or enter debug mode")
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=False, 
        default=None,
        help="Config file. If not provided, defaults will be used."
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
    
    return config, args.debug

def set_config_defaults(config):
    config = {
        "seed": config.get("seed", 42),
        "run_name": config.get("run_name", "run"),

        # Model and training details
        "model_type": config.get("model_type", "fast_cnn"), # "fast_cnn", "fast_cnn2", "flex_net", "custom_vit", "resnet18", "zoom",
        "optimizer": config.get("optimizer", "adam"),
        "scheduler": config.get("scheduler", "CosineAnnealingLR"), # ReduceLROnPlateau or CosineAnnealingLR
        "scheduler_patience": config.get("scheduler_patience", 3), # applies only to ReduceLROnPlateau
        "scheduler_T_max": config.get("scheduler_T_max", 50), # # applies only to CosineAnnealingLR
        "criterion": config.get("criterion", "CE"),
        "label_smoothing": config.get("label_smoothing", 0.1), # smoothing for criterion
        "train_split": config.get("train_split", 0.9),
        "batch_size": config.get("batch_size", 256),
        "num_workers": config.get("num_workers", 1),
        "lr": config.get("lr", 0.001),
        "epochs": config.get("epochs", 1),
        "ema": config.get("ema", False), # Exponential Moving Average
    
        # Early stopping
        "early_stopping": config.get("early_stopping", True),
        "min_diff": config.get("min_diff", 0.001),
        "writer": config.get("writer", True),
        "patience": config.get("patience", 5),

        # BrainIt specific config
        "retinal_layer": config.get("retinal_layer", True),

        # CustomVit specific config
        "use_flex": config.get("use_flex", False),

        # ZoomViT-specific configs
        "use_pos_embed": config.get("use_pos_embed", False),
        "add_dropout": config.get("add_dropout", False),
        "mlp_end": config.get("mlp_end", False),
        "add_cls_token": config.get("add_cls_token", False),
        "num_layers": config.get("num_layers", 2),
        "trans_dropout_ratio": config.get("trans_dropout_ratio", 0.0),
        "standard_scale": config.get("standard_scale", False),
        "resnet_layers": config.get("resnet_layers", 4),
        "multiscale_tokenisation": config.get("multiscale_tokenisation", False),
        "gamma_per_head": config.get("gamma_per_head", False),
        "use_token_mixer": config.get("use_token_mixer", False),
        "remove_zoom": config.get("remove_zoom", False),

        # Data
        "transform_type": config.get("transform_type", "custom"), # "custom", "custom_agg", "default"
        "downsample_fraction": config.get("downsample_fraction", 0),
        "few_shot": config.get("few_shot", False),
        "freeze_resnet_early": config.get("freeze_resnet_early", False),

        "attacker": config.get("attacker", False), # FGSM, PGD, False
        "epsilon": config.get("epsilon", False), # epsilon 0 - no attack

        "gaussian_std": config.get("gaussian_std", False),
    }
    return config

def main():
    try:
        config, debug = parse_args()
        config = set_config_defaults(config)

        time_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = get_log_dir(config["run_name"], time_stamp)
        logger = get_logger(log_dir)
        logger.info("Job Started")
        logger.info(f"Logging run: {log_dir}")

        save_conifg_path = os.path.join(log_dir, f"config_{config['run_name']}.json")
        with open(save_conifg_path, 'w') as f:
            f.write(json.dumps(config, indent=2))

        valid_models = [
            "fast_cnn",
            "fast_cnn2",
            "flex_net",
            "custom_vit",
            "resnet18",
            "zoom",
            "brainit"
        ]
        assert config["model_type"] in valid_models, "Invalid model type"
        assert config["optimizer"] in ["adam"], "Invalid optimizer"
        assert config["criterion"] in ["CE"], "Invalid criterion"
        assert config["scheduler"] in ["CosineAnnealingLR", "ReduceLROnPlateau"], "Invalid scheduler"
        assert config["attacker"] in [False, "FGSM", "PGD"], "Invalid attacker"
        assert (config["attacker"] and config["epsilon"]) or not config["attacker"], "Invalid"
        assert (config["gaussian_std"] and isinstance(config["gaussian_std"], float)) or not config["gaussian_std"], "Invalid"

        if config["few_shot"] and config["downsample_fraction"] > 0:
            logger.info("Few shot and downsampling fraction provided! Only doing downsampling.")
        
        if config["add_cls_token"] and not config["use_pos_embed"]:
            logger.info("CLS token being used. Setting use_pos_embed to true!")
            config["use_pos_embed"] = True

        if config["writer"]:
            writer = SummaryWriter(log_dir=log_dir)
            writer.add_text("config", json.dumps(config, indent=2))

        g = set_seed(config["seed"])
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Using device: {device}")
        logger.info(f"Training for {config['epochs']} epochs")
        logger.info(f"Debug mode: {debug}")
        logger.info(f"Downsampling by: {config['downsample_fraction']}")

        norm_means, norm_stds = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        transform_train = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_means, norm_stds)])
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(norm_means, norm_stds)])
        custom_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_stds)
        ])
        custom_transform_agg = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_stds)
        ])
        foveation_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            FixedFoveation(),
            transforms.Normalize(norm_means, norm_stds)
        ])
        if config["model_type"] == "fast_cnn":
            model = FastCNN().to(device)
        elif config["model_type"] == "fast_cnn2":
            model = FastCNN2().to(device)
        elif config["model_type"] == "flex_net":
            model = FlexNet(device=device).to(device)
        elif config["model_type"] == "custom_vit":
            model = ConvViTHybrid(device=device, use_flex=config["use_flex"]).to(device)
        elif config["model_type"] == "resnet18":
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
                standard_scale=config["standard_scale"],
                resnet_layers=config["resnet_layers"],
                multiscale_tokenisation=config["multiscale_tokenisation"],
                freeze_resnet_early=config["freeze_resnet_early"],
                gamma_per_head=config["gamma_per_head"],
                use_token_mixer=config["use_token_mixer"],
                remove_zoom=config["remove_zoom"]
            ).to(device)
        elif config["model_type"] == "brainit":
            model = BrainiT(
                use_retinal_layer=config["retinal_layer"],
                device=device
            ).to(device)
        
        if config["transform_type"] == "custom":
            logger.info("Using custom transform")
            transform_train = custom_transform
        elif config["transform_type"] == "custom_agg":
            logger.info("Using custom agg transform")
            transform_train = custom_transform_agg
        elif config["transform_type"] == "foveation":
            logger.info("Using foveation transform")
            transform_train = foveation_transform
            transform_test = transforms.Compose([
                transforms.ToTensor(), 
                FixedFoveation(),
                transforms.Normalize(norm_means, norm_stds)
            ])

        train_loader, val_loader, test_loader = load_data(
            transform_train,
            transform_test,
            config["train_split"],
            config["batch_size"],
            config["num_workers"],
            seed_worker,
            g,
            downsample_fraction=config["downsample_fraction"],
            few_shot=config["few_shot"],
            debug=debug,
            logger=logger,
            test_batch_size=32 if config["attacker"] else config["batch_size"]
        )

        # input_size = (32, 32)
        # summary_str = summary(model, input_size=(1, 3, input_size[0], input_size[1]), device=device, verbose=0)
        # with open(os.path.join(log_dir, "model_summary.txt"), "w") as f:
        #     f.write(str(summary_str))
    
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
        
        if config["scheduler"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config["scheduler_patience"])
        elif config["scheduler"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(optimizer, T_max=config["scheduler_T_max"], eta_min=1e-5)
        
        if config["criterion"] == "CE":
            logger.info(f"Label smoothing: {config['label_smoothing']}")
            criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])

        if config["early_stopping"]:
            early_stopping = EarlyStopping(config["patience"], config["min_diff"])
        
        if config["ema"]:
            logger.info("Using EMA")
            ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        
        if config["attacker"] == "FGSM":
            attacker = torchattacks.FGSM(model, eps=config["epsilon"])
        elif config["attacker"] == "PGD":
            attacker = torchattacks.PGD(model, eps=config["epsilon"], alpha=2/255, steps=10)

        best_loss_model, last_model, train_losses, val_losses, train_acc, val_acc = train_model(
            model, 
            config["epochs"] if not debug else 1, 
            train_loader, 
            val_loader,
            device,
            optimizer,
            criterion,
            scheduler,
            early_stopping if config["early_stopping"] else False,
            ema if config["ema"] else False,
            writer if config["writer"] else False,
            logger
        )
        torch.save(best_loss_model["model_state"], f"{log_dir}/{config['run_name']}_{time_stamp}_e{best_loss_model['epoch']}_best_model.pt")
        logger.info(f"Best model with lowest val loss {best_loss_model['val_loss']} at {best_loss_model['epoch']} epoch is saved.")
        model.load_state_dict(best_loss_model["model_state"])
        best_loss_acc, best_loss_cls_report = test_model(
            model, 
            test_loader, 
            device,
            writer if config["writer"] else False, 
            logger, 
            attacker if config["attacker"] else False, 
            config["gaussian_std"]
        )

        torch.save(last_model["model_state"], f"{log_dir}/{config['run_name']}_{time_stamp}_e{last_model['epoch']}_last_model.pt")
        logger.info(f"Last model with val loss {last_model['val_loss']} at {last_model['epoch']} epoch is saved.")
        model.load_state_dict(last_model["model_state"])
        last_acc, last_cls_report = test_model(
            model,
            test_loader,
            device,
            writer if config["writer"] else False,
            logger,
            attacker if config["attacker"] else False,
            config["gaussian_std"]
        )

        metric_file_pt = os.path.join(log_dir, f"metrics_{config['run_name']}_{time_stamp}.csv")
        with open(metric_file_pt, "w+") as f:
            f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")
            f.write(f"Best model test acc and loss at epoch {best_loss_model['epoch']}:,{best_loss_acc},0,,\n")
            f.write(f"Last model test acc and loss at epoch {last_model['epoch']}:,{last_acc},0,,\n")
            for i in range(len(train_losses)):
                f.write(f"{i},{train_losses[i]},{val_losses[i]},{train_acc[i]},{val_acc[i]}\n")
        
        best_cls_report_pt = os.path.join(log_dir, f"cls_report_best_loss_{config['run_name']}_{time_stamp}.txt")
        last_cls_report_pt = os.path.join(log_dir, f"cls_report_last_{config['run_name']}_{time_stamp}.txt")
        with open(best_cls_report_pt, "w") as f:
            f.write(best_loss_cls_report)
        with open(last_cls_report_pt, "w") as f:
            f.write(last_cls_report)

        if writer:
            writer.close()
        logger.info("\nJob Completed")

    except Exception as e:
        logger.exception("Error occured")
        raise
    

if __name__ == "__main__":
    main()
