import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import torchvision

def cifar_class_balanced_subset(dataset, fraction):
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    indices = []

    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        selected = np.random.choice(cls_idx, size=int(len(cls_idx) * fraction), replace=False)
        indices.extend(selected)

    np.random.shuffle(indices) 
    return Subset(dataset, indices)

def cifar_few_shot_subset(dataset, n_per_class):
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    indices = []

    for cls in classes:
        cls_idx = np.where(targets == cls)[0]
        selected = np.random.choice(cls_idx, size=n_per_class, replace=False)
        indices.extend(selected)

    np.random.shuffle(indices)
    return Subset(dataset, indices)

def load_cifar10(
    dataset_configs,
    transform_train,
    transform_test,
    seed_worker_fn,
    seed_generator,
    logger,
    test_batch_size=None,
    debug=False
):
    batch_size = dataset_configs["batch_size"]
    downsample_fraction = dataset_configs["downsample_fraction"]
    few_shot = dataset_configs["few_shot"]
    train_split = dataset_configs["train_split"]
    num_workers = dataset_configs["num_workers"]
    if not test_batch_size:
        test_batch_size = batch_size

    full_train_set = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=False, download=True, transform=transform_test)
   
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
        full_train_set = cifar_class_balanced_subset(full_train_set, downsample_fraction)
    elif few_shot:
        full_train_set = cifar_few_shot_subset(full_train_set, few_shot)

    logger.info(f"Full train/val size with downsampling ({downsample_fraction}): {len(full_train_set)}")

    train_size = int(train_split * len(full_train_set))
    val_size = len(full_train_set) - train_size
    train_set, val_set = random_split(full_train_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker_fn, generator=seed_generator)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker_fn, generator=seed_generator)
    test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker_fn, generator=seed_generator)

    return train_loader, val_loader, test_loader
