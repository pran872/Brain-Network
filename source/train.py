'''Simple CNN'''
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import torch.nn as nn
import torchattacks
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from timm.scheduler import CosineLRScheduler
from timm.loss import SoftTargetCrossEntropy
from torch_ema import ExponentialMovingAverage
from torchinfo import summary
from contextlib import nullcontext
from sklearn.metrics import classification_report
from tqdm import tqdm
import time
import numpy as np
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
import socket
from collections import defaultdict

try:
    from models import *
    from transform import get_transform
    from datasets.get_dataset import get_data
    from utils import *
except ModuleNotFoundError:
    from source.models import *
    from source.transform import get_transform
    from source.datasets.get_dataset import get_data
    from source.utils import *

def get_log_dir(run_name: str, time_stamp, base_env_var="OUTPUT_DIR") -> str:
    full_run_name = f"{run_name}_{time_stamp}"

    output_root = os.environ.get(base_env_var)

    if not output_root:
        hostname = socket.gethostname()
        if "login" in hostname or "node" in hostname or "rds" in os.getcwd():
            output_root = "/rds/general/user/psp20/home/Brain-Network/runs"
        else:
            output_root = "runs/"

    log_dir = os.path.join(output_root, full_run_name)
    os.makedirs(log_dir, exist_ok=True)

    return log_dir
    
def train_model(
    model,
    epochs,
    train_loader,
    val_loader,
    device,
    optimizer,
    criterion,
    auxiliary_loss,
    scheduler,
    early_stopping,
    ema,
    mixup_fn,
    writer,
    logger
):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_model = {"val_loss": float("inf"),
                  "model_state": None,
                  "epoch": 0}
    gamma_by_class = defaultdict(list)
    logger.info(f"Auxiliary loss: {auxiliary_loss}")

    for epoch in range(epochs):
        start = time.time()
        gamma_loss = False
        model.train()
        total_loss, correct, total, total_gamma_loss = 0, 0, 0, 0

        for batch_idx, (images, labels) in tqdm(enumerate(train_loader), desc=f"Epoch {epoch+1} [Train]", disable=not sys.stdout.isatty()):
            images, labels = images.to(device), labels.to(device) # [B, C, 224, 224], [B]

            if mixup_fn is not None:
                images, labels = mixup_fn(images, labels) # [B, C, 224, 224], [B, 120]

            optimizer.zero_grad()
            if isinstance(model, ZoomVisionTransformer):
                outputs, gamma, attn_map = model(images, return_gamma=True, return_attn_map=True)
                gamma_loss = compute_auxiliary_loss(auxiliary_loss, gamma, attn_map)
            else:
                outputs = model(images)

            loss = criterion(outputs, labels)
            if gamma_loss:
                loss += gamma_loss
                total_gamma_loss += gamma_loss.item()
            loss.backward()

            if writer and batch_idx == 0: 
                log_grad(writer, model, epoch)

            optimizer.step()
            if ema:
                ema.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)

            if mixup_fn is not None:  # [B, num_classes], mixup active
                targets = labels.argmax(dim=1)
            else: 
                targets = labels

            correct += predicted.eq(targets).sum().item()

        avg_train_loss = total_loss / len(train_loader)
        if gamma_loss:
            avg_train_gamma_loss = total_gamma_loss / len(train_loader)
        train_acc = 100. * correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)

        # Validation 
        val_criterion = torch.nn.CrossEntropyLoss()
        if ema:
            context = ema.average_parameters()
        else:
            context = nullcontext()
        
        with context:
            model.eval()
            gamma_loss = False
            val_loss, val_correct, val_total, val_gamma_loss = 0, 0, 0, 0
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    if isinstance(model, FlexNet):
                        outputs, conv_ratio = model(images, return_conv_ratio=True)

                    elif isinstance(model, ZoomVisionTransformer):
                        if isinstance(model, (BrainiT, BrainiT224)):
                            outputs, gamma, attn_map, cx, cy, = model(images, return_cx_cy=True, return_gamma=True, return_attn_map=True)
                        elif isinstance(model, ZoomVisionTransformer):
                            outputs, gamma, attn_map = model(images, return_gamma=True, return_attn_map=True)
                        gamma_by_class = process_gamma(gamma, gamma_by_class, labels)
                        gamma_loss = compute_auxiliary_loss(auxiliary_loss, gamma, attn_map)

                    else:
                        outputs = model(images)

                    loss = val_criterion(outputs, labels)
                    if gamma_loss is not False:
                        loss += gamma_loss
                        val_gamma_loss += gamma_loss

                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        if gamma_loss is not False:
            avg_val_gamma_loss = val_gamma_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_acc)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineLRScheduler):
            scheduler.step(epoch)
    
        end = time.time()

        if writer:
            writer.add_scalar("Loss/train", avg_train_loss, epoch)
            writer.add_scalar("Loss/val", avg_val_loss, epoch)
            writer.add_scalar("Accuracy/train", train_acc, epoch)
            writer.add_scalar("Accuracy/val", val_acc, epoch)
            if gamma_loss is not False:
                writer.add_scalar("GammaLoss/train", avg_train_gamma_loss, epoch)
                writer.add_scalar("GammaLoss/val", avg_val_gamma_loss, epoch)

            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar("LR", current_lr, epoch)

            if isinstance(model, FlexNet):
                writer.add_scalar("Conv_ratio", conv_ratio, epoch)
            if isinstance(model, ZoomVisionTransformer):
                writer.add_scalar("Gamma/mean", gamma.mean().item(), epoch)
                writer.add_scalar("Gamma/std", gamma.std().item(), epoch)
                for c in range(len(gamma_by_class)):
                    writer.add_scalar(f"Gamma/Class_{c}", np.mean(gamma_by_class[c]), epoch)
            if isinstance(model, (BrainiT, BrainiT224)):
                writer.add_scalar("Centroid_means/cx", cx, epoch)
                writer.add_scalar("Centroid_means/cy", cy, epoch)

        logger.info(f"Epoch: {epoch+1} | Time: {end - start:.2f}s")
        logger.info(f"Train Loss={avg_train_loss:.4f}, Train Acc={train_acc:.2f}%")
        logger.info(f"Val Loss={avg_val_loss:.4f}, Val Acc={val_acc:.2f}%")
        if gamma_loss:
            logger.info(f"Train Gamma Loss={avg_train_gamma_loss:.4f} | Val Gamma Loss={avg_val_gamma_loss}")
        
        if avg_val_loss < best_model["val_loss"]:
            best_model["val_loss"] = avg_val_loss
            best_model["model_state"] = model.state_dict()
            best_model["epoch"] = epoch
            logger.info(f"New best model saved (val loss = {best_model['val_loss']:.4f})")

        if early_stopping and isinstance(early_stopping, EarlyStopping):
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

def test_model(model,
    test_loader,
    device,
    writer,
    logger,
    attacker=False,
    gaussian_std=False
):
    if attacker:
        logger.info(f"Using attacker: {attacker}")

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

        if isinstance(outputs, (tuple, list)): 
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

def set_config_defaults(user_config):
    with open("configs/config_template.json", 'r') as f:
        config = json.load(f) # default_config

    for key in config:
        if key in user_config:
            config[key] = user_config[key]

    return config

def main(parsed_args=None):
    try:
        if parsed_args is None:
            config, debug = parse_args()
        else:
            config, debug = parsed_args

        config = set_config_defaults(config)

        time_stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = get_log_dir(config["run_name"], time_stamp)
        logger = get_logger(log_dir)
        logger.info("Job Started")
        logger.info(f"Logging run: {log_dir}")

        save_config_path = os.path.join(log_dir, f"config_{config['run_name']}.json")
        with open(save_config_path, 'w') as f:
            f.write(json.dumps(config, indent=2))

        valid_models = [
            "fast_cnn",
            "fast_cnn2",
            "flex_net",
            "custom_vit",
            "resnet18",
            "zoom",
            "zoom224",
            "brainit"
        ]
        assert config["dataset"]["type"] in ["cifar10", "stanford_dogs"], "Invalid dataset"
        assert config["model_type"] in valid_models, "Invalid model type"
        assert config["optimizer"] in ["adam", "adamW"], "Invalid optimizer"
        assert config["criterion"] in ["CE"], "Invalid criterion"
        assert list(config["scheduler"].keys())[0] in ["CosineAnnealingLR", "ReduceLROnPlateau", "CosineLRScheduler"], "Invalid scheduler"
        assert config["attacker"] in [False, "FGSM", "PGD"], "Invalid attacker"
        assert (config["attacker"] and config["epsilon"]) or not config["attacker"], "Invalid"
        assert (config["gaussian_std"] and isinstance(config["gaussian_std"], float)) or not config["gaussian_std"], "Invalid"

        if config["dataset"]["few_shot"] and config["dataset"]["downsample_fraction"] > 0:
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
        logger.info(f"Downsampling by: {config['dataset']['downsample_fraction']}")
        logger.info(f"Auxiliary loss: {config['auxiliary_loss']}")

        transform_train, transform_test = get_transform(config["dataset"]["type"], config["transforms"], pretrained=config["pretrained"])
        logger.info(f"Train transforms: \n\t{transform_train}")
        logger.info(f"Test transforms: \n\t{transform_test}")
        train_loader, val_loader, test_loader = get_data(
            config["dataset"],
            transform_train,
            transform_test,
            seed_worker,
            g,
            logger,
            test_batch_size=32 if config["attacker"] else None,
            debug=debug,
        )

        if config["pretrained"]:
            logger.info(f"Using pretrained weights")
        if config["freeze_all"]:
            logger.info(f"Freezing Resnet")


        model = get_model(config, device)
        model.to(device)
    
        if config["optimizer"] == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
        elif config["optimizer"] == "adamW":
            weight_decay = 1e-4 if config["model_type"] in ["resnet18", "fast_cnn2"] else 1e-2
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=weight_decay)
        
        if "ReduceLROnPlateau" in config["scheduler"].keys():
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=config["scheduler_patience"])
        elif "CosineAnnealingLR" in config["scheduler"].keys():
            scheduler = CosineAnnealingLR(optimizer, T_max=config["scheduler"]["CosineAnnealingLR"]["scheduler_T_max"], eta_min=1e-5)
        elif "CosineLRScheduler" in config["scheduler"].keys():
            warmup_t = config["scheduler"]["CosineLRScheduler"].get("warmup_t", 5)
            scheduler = CosineLRScheduler(
                optimizer,
                t_initial=config["epochs"] - warmup_t,
                warmup_t=warmup_t,
                warmup_lr_init=1e-5,
                lr_min=1e-6,
            )
        
        if config["mixup_fn"]:
            num_classes = 120 if config["dataset"]["type"] == "stanford_dogs" else 10
            mixup_fn = get_mixup_fn(num_classes)
            logger.info("Using mixup function and SoftTargetCrossEntropy")
            criterion = SoftTargetCrossEntropy()
        else:   
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
            config["auxiliary_loss"],
            scheduler,
            early_stopping if config["early_stopping"] else False,
            ema if config["ema"] else False,
            mixup_fn if config["mixup_fn"] else None,
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
    
    return log_dir
    

if __name__ == "__main__":
    main()
