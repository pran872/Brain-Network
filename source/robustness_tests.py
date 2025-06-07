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


def test_model(
    model,
    test_loader,
    device
):
    model.eval()
    correct, total = 0, 0
    y_pred, y_true = [], []
    for images, labels in tqdm(test_loader, desc="Testing", disable=not sys.stdout.isatty()):
        images, labels = images.to(device), labels.to(device)

        with torch.no_grad():
            outputs = model(images)

        if isinstance(outputs, tuple): 
            outputs = outputs[0]

        _, predicted = outputs.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        y_pred.append(predicted.cpu().numpy())
        y_true.append(labels.cpu().numpy())

    
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
        
        config_file = glob.glob(os.path.join(args.run_folder, "*.json"))
        if len(config_file) == 0:
            logger.info("No config present. Exiting")
        assert len(config_file) > 0, "No config file present in the provided directory."

        if len(config_file) > 1:
            logger.info(f"The provided directory has more than one config file. Using {os.path.basename(config_file[0])}.")
        args.config = config_file[0]

        args.model = glob.glob(os.path.join(args.run_folder, "*best*.pt"))
        if len(args.model) == 0:
            logger.info("No model present. Exiting")
        assert len(args.model) > 0, "No model path present in the provided directory."

        if len(args.model) > 1:
            logger.info(f"The provided directory has more than one config file. Using {os.path.basename(args.model[0])}")
        args.model = args.model[0]
    
    with open(args.config, 'r') as f:
        config = json.load(f)
        
    model_pt = args.model
    
    return config, args.config, model_pt

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
        "--run_all_attacks",
        action="store_true",
        required=False,
        help="Runs all IMAGENET-C perturbations"
    )
    parser.add_argument(
       "--corruption", 
        type=str, 
        required=False, 
        default="brightness",
        help="Corruption to use (no .npy)"
    )
    parser.add_argument(
       "--severity", 
        type=str, 
        required=False, 
        default="1",
        help="Severity to use"
    )

    args = parser.parse_args()
    return args, args.debug, args.run_all_attacks

def main(parsed_args=None):
    if parsed_args is None:
        args, debug, run_all_attacks = parse_args()
    else:
        args, debug, run_all_attacks = parsed_args
        
    log_dir = args.run_folder if args.run_folder else os.path.dirname(args.config)
    logger = get_logger(log_dir)
    logger.info("")
    logger.info("Running robustness tests")

    config, config_pt, model_pt = get_files(args, logger)
    config = set_config_defaults(config)

    logger.info(f"Using config: {config_pt}")
    logger.info(f"Using model: {model_pt}")
    
    assert config["model_type"] in ["zoom", "resnet18", "brainit"], "Testing only supports zoom, resnet18, and brainit models"

    g = set_seed(config["seed"])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform_train, transform_test = get_transform(config["dataset"]["type"], config["transforms"], pretrained=config["pretrained"])

    if debug:
        all_corruptions = [
            "brightness", "contrast"
        ]
        all_severities = [1, 2]
    elif run_all_attacks:
        all_corruptions = [
            "brightness", "contrast", "defocus_blur", "elastic_transform", "fog", "frost", "gaussian_blur", "gaussian_noise", "glass_blur", "impulse_noise", "jpeg_compression", "motion_blur", "pixelate", "saturate", "shot_noise", "snow", "spatter", "speckle_noise", "zoom_blur"
        ]
        all_severities = [1, 2, 3, 4, 5]
    else:
        all_corruptions = [args.corruption]
        all_severities = [args.severity]
    
    logger.info(f"Running these corruptions: {all_corruptions}")
    logger.info(f"For these severities: {all_severities}")

    config["batch_size"] = 32

    results = {}
    for corruption in all_corruptions:
        results[corruption] = []
        logger.info(f"Corruption type: {corruption}")

        for severity in all_severities:
            logger.info(f"Severity type: {severity}")

            config["dataset"]["corruption"] = corruption
            config["dataset"]["severity"] = severity

            _, _, test_loader = get_data(
                config["dataset"],
                transform_train,
                transform_test,
                seed_worker,
                g,
                logger,
                test_batch_size=32,
                robustness_testing=True,
                debug=debug,
            )

            model = load_model(model_pt, config, device, logger)
            acc, cls_report = test_model(
                model,
                test_loader,
                device
            )
            
            logger.info(f"Test accuracy: {acc}")
            
            results[corruption].append(acc)
    
    log_dir = os.path.dirname(model_pt)
    with open(f'{log_dir}/robustness_results.txt', 'w') as file:
        for key, value in results.items():
            file.write(f'{key}: {value}\n')
    

if __name__ == "__main__":
    main()
