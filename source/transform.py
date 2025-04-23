import torchvision.transforms as transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy

try:
    from brainit import *
except ModuleNotFoundError:
    from source.brainit import *

def get_transform(dataset_type, transform_type):
    if dataset_type == "cifar10":    
        norm_means, norm_stds = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        transform_test_list = [transforms.ToTensor(), transforms.Normalize(norm_means, norm_stds)]

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
        norm_means, norm_stds = [0.4765, 0.4517, 0.3911], [0.2264, 0.2214, 0.2193]

        transform_train_list = [
            transforms.Resize((224, 224), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_stds)
        ]
        transform_test_list = [
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(norm_means, norm_stds)
        ]

    transform_train = transforms.Compose(transform_train_list)
    transform_test = transforms.Compose(transform_test_list)
    
    return transform_train, transform_test