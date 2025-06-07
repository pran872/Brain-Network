import numpy as np
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

class CIFAR10CCorruption(Dataset):
    def __init__(self, corruption_name, severity_level, transform):
        self.images = np.load(f"data/robustness_data/CIFAR-10-C/{corruption_name}.npy")
        self.labels = np.load(f"data/robustness_data/CIFAR-10-C/labels.npy")

        assert 1 <= severity_level <= 5, "Severity must be between 1 and 5"
        start = (severity_level - 1) * 10000
        end = severity_level * 10000

        self.images = self.images[start:end]
        self.labels = self.labels[start:end]

        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        
        return img, label


def load_robustness_data(
    dataset_configs,
    transform_test,
    seed_worker_fn,
    seed_generator,
    logger=None,
    test_batch_size=None,
    debug=False
):
    corruption_name = dataset_configs["corruption"]
    severity_level = dataset_configs["severity"]
    batch_size = dataset_configs["batch_size"]
    num_workers = dataset_configs["num_workers"]

    if not test_batch_size:
        test_batch_size = batch_size

    test_dataset = CIFAR10CCorruption(
        corruption_name=corruption_name,
        severity_level=severity_level,
        transform=transform_test
    )
   
    if debug:
        test_dataset = Subset(test_dataset, list(range(200)))
        if logger:
            logger.info("DEBUG mode: using 200 robustness samples")
    else:
        if logger:
            logger.info(f"FULL mode: using {len(test_dataset)} robustness samples")

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker_fn,
        generator=seed_generator
    )
    
    return None, None, test_loader
