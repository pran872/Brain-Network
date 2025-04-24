import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
import copy

try:
    from brainit import *
    from constants import NORMS
except ModuleNotFoundError:
    from source.brainit import *
    from source.constants import NORMS

def get_transform(dataset_type, input_transforms, pretrained=False):
    if pretrained:
        norm_means, norm_stds = NORMS["imagenet"]
    else:
        norm_means, norm_stds = NORMS[dataset_type]

    if dataset_type == "cifar10":    
        transform_test_list = [transforms.ToTensor(), transforms.Normalize(norm_means, norm_stds)]
        transform_type = input_transforms["transform_type"]

        if transform_type == "custom":
            transform_train_list = [
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_means, norm_stds)
            ]
        elif transform_type == "custom_agg":
            transform_train_list =[
                transforms.RandomCrop(32, padding=4),
                transforms.AutoAugment(policy=AutoAugmentPolicy.CIFAR10),
                transforms.ToTensor(),
                transforms.Normalize(norm_means, norm_stds)
            ]
        elif transform_type == "foveation":
            transform_train_list = transforms[
                transforms.RandomCrop(32, padding=4), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                FixedFoveation(),
                transforms.Normalize(norm_means, norm_stds)
            ]
            transform_test_list = transforms[
                transforms.ToTensor(), 
                FixedFoveation(),
                transforms.Normalize(norm_means, norm_stds)
            ]
        else:
            transform_train_list = transform_test_list

        transform_train = transforms.Compose(transform_train_list)
        transform_test = transforms.Compose(transform_test_list)

    elif dataset_type == "stanford_dogs":

        transform_train_list = build_transforms(input_transforms["train_transforms"])
        transform_test_list = build_transforms(input_transforms["test_transforms"])

        add_ons = [
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_stds)
        ]
        transform_train_list.extend(add_ons)
        transform_test_list.extend(add_ons)

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)
    
    return transform_train, transform_test

def build_transforms(transform_config_list):
    transform_dict = {
        "RRC": lambda **kwargs: transforms.RandomResizedCrop(**kwargs),
        "resize": lambda **kwargs: transforms.Resize(**kwargs, antialias=True),
        "center_crop": lambda **kwargs: transforms.CenterCrop(**kwargs),
        "horizontal_flip": lambda **kwargs: transforms.RandomHorizontalFlip(**kwargs),
        "color_jitter": lambda **kwargs: transforms.ColorJitter(**kwargs),
    }

    transform_list = []
    for specific_trans in transform_config_list:
        if isinstance(specific_trans, str):
            trans = transform_dict[specific_trans]()
        elif isinstance(specific_trans, dict):
            name, params = list(specific_trans.items())[0] # e.g., ('RRC', {'size': 224, 'scale': [0, 1]})
            trans = transform_dict[name](**params)
        
        transform_list.append(trans)

    return transform_list
