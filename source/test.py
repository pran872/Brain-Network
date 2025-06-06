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
import glob
try:
    from models import *
    from train import set_config_defaults
    from transform import get_transform
    from datasets.get_dataset import get_data
    from brainit import FixedFoveation
    from utils import *
except ModuleNotFoundError:
    from source.models import *
    from source.train import set_config_defaults
    from source.datasets.get_dataset import get_data
    from source.brainit import FixedFoveation
    from source.transform import get_transform
    from source.utils import *
import os
import sys

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
    cls_report = classification_report(y_true, y_pred, output_dict=False)

    return test_acc, cls_report

def load_model(model_pt, config, device, logger):
    model = get_model(config, device)
    model.to(device)

    checkpoint = torch.load(model_pt, map_location=device)
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError:
        if config["dataset"]["type"] == "cifar10":
            logger.info("Runtime error occurred. Using old cifar10 models.")
            model = get_model(config, device, load_old_models=True)
            model.to(device)
            checkpoint = torch.load(model_pt, map_location=device)
            model.load_state_dict(checkpoint)

        else:
            raise RuntimeError
    return model

def get_files(args, logger):
    assert (args.model and args.config) or args.run_folder, "Please provide either (run_directory) or (model and config paths)"
    
    if not args.model and not args.config and args.run_folder:
        config_file = glob.glob(os.path.join(args.run_folder, "**", "*test*.json"), recursive=True)
        if len(config_file) == 0:
            logger.info("No test config present. Using train config.")
            config_file = glob.glob(os.path.join(args.run_folder, "**", "*.json"), recursive=True)
        assert len(config_file) > 0, "No config file present in the provided directory."

        if len(config_file) > 1:
            logger.info(f"The provided directory has more than one config file. Using {os.path.basename(config_file[0])}.")
        args.config = config_file[0]

        args.model = glob.glob(os.path.join(args.run_folder, "**", "*test*best*.pt"), recursive=True)
        assert len(args.model) > 0, "No model path present in the provided directory."
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    assert all([model_pt.endswith(".pt") for model_pt in args.model])
    model_pts = args.model
    
    return config, model_pts

def parse_args():
    parser = argparse.ArgumentParser(description="Pass config")
    parser.add_argument(
        "--run_folder",
        type=str,
        required=False,
        default=None,
        help="Pass the path to the run. If not provided, pass config and model paths."
    )
    parser.add_argument(
        "-c", "--config", 
        type=str, 
        required=False, 
        default=None,
        help="Config file. If not provided, defaults will be used."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=False,
        default=None
    )
    parser.add_argument(
        "-d", "--debug", 
        action="store_true",
        required=False,
        help="Run on debug mode"
    )
    parser.add_argument(
        "--run_default_attacks",
        action="store_true",
        required=False,
        help="Runs FGSM, PGD, and gaussian noise with epsilon [0.01, 0.05, 0.1, 0.2]"
    )

    args = parser.parse_args()
    return args, args.debug, args.run_default_attacks

def main(parsed_args=None):
    if parsed_args is None:
        args, debug, run_default_attacks = parse_args()
    else:
        args, debug, run_default_attacks = parsed_args
        
    log_dir = args.run_folder if args.run_folder else os.path.dirname(args.config)
    logger = get_logger(log_dir)

    config, model_pts = get_files(args, logger)
    config = set_config_defaults(config)
    
    assert config["attacker"] in [False, "FGSM", "PGD"], "Invalid attacker"
    assert (config["attacker"] and config["epsilon"]) or not config["attacker"], "Invalid"
    assert (config["gaussian_std"] and isinstance(config["gaussian_std"], float)) or not config["gaussian_std"], "Invalid"
    assert config["model_type"] in ["zoom", "resnet18", "brainit"], "Testing only supports zoom, resnet18, and brainit models"

    if config["attacker"] and config["gaussian_std"]:
        logger.info("Both attacker and gaussian std passed. Performing only attack, not gaussian noise")
        config["gaussian_std"] = False

    g = set_seed(config["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train, transform_test = get_transform(config["dataset"]["type"], config["transforms"], pretrained=config["pretrained"])
    
    _, _, test_loader = get_data(
        config["dataset"],
        transform_train,
        transform_test,
        seed_worker,
        g,
        logger,
        test_batch_size=32 if config["attacker"] or run_default_attacks else None,
        debug=debug,
    )

    config["batch_size"] = 32 if config["attacker"] or run_default_attacks else config["dataset"]["batch_size"]
    logger.info(f"Batch size: {config['dataset']['batch_size']}")

    logger.info("Running these attacks:")
    if run_default_attacks:
        epsilons = [0.01, 0.05, 0.1, 0.2] if not debug else [0.01]
        fgsm_attacks = {"FGSM": epsilons}
        pgd_attacks = {"PGD": epsilons}
        gaussian_noise = {"gaussian noise": epsilons}
        all_attacks = fgsm_attacks | pgd_attacks | gaussian_noise # concat dicts
        #all_attacks = pgd_attacks
    elif config["attacker"] == "FGSM":
        all_attacks = {"FGSM": config["epsilon"]}
    elif config["attacker"] == "PGD":
        all_attacks = {"PGD": config["epsilon"]}
    else:
        all_attacks = {"None": [False]}
    
    logger.info(all_attacks)
    
    for model_pt in model_pts:
        logger.info(f"\nTesting model: {model_pt}")

        for attack_type, all_epsilons in all_attacks.items():
            for epsilon in all_epsilons:
                logger.info(f"Running attack: {attack_type}, {epsilon}")

                model = load_model(model_pt, config, device, logger)

                gaussian_std = False
                attacker = False
                if attack_type == "FGSM":
                    attacker = torchattacks.FGSM(model, eps=epsilon)
                elif attack_type == "PGD":
                    attacker = torchattacks.PGD(model, eps=epsilon, alpha=2/255, steps=10)
                elif attack_type == "gaussian noise":
                    gaussian_std = epsilon
                
                acc, cls_report = test_model(
                    model,
                    test_loader,
                    device,
                    attacker,
                    gaussian_std
                )
                log_dir = os.path.dirname(model_pt)
                best_or_last = os.path.basename(model_pt).split("_")[-2]
                str_epsilon = str(epsilon).replace(".", "_")
                if debug:
                    results_fname = f"{log_dir}/test_manual_debug_{attack_type}_{str_epsilon}_{best_or_last}_{config['run_name']}.txt"
                else:
                    results_fname = f"{log_dir}/test_manual_{attack_type}_{str_epsilon}_{best_or_last}_{config['run_name']}.txt"
                with open(results_fname, "w") as f:
                    f.write(f"Test accuracy: {acc}")
                    f.write(cls_report)
                logger.info(f"Test accuracy: {acc}")
        break

if __name__ == "__main__":
    main()
