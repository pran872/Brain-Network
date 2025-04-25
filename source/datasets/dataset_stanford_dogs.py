'''Stanford dogs: http://vision.stanford.edu/aditya86/ImageNetDogs/'''

import scipy.io
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import defaultdict
import random
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(root_dir)

def load_img_paths(mat_path):
    mat = scipy.io.loadmat(mat_path)
    root_img_dir = "./data/stanford_dogs/Images"
    img_list = [
        (os.path.join(root_img_dir, mat['file_list'][i][0][0]), int(mat['labels'][i][0])-1) 
        for i in range(len(mat['file_list']))
    ]
    return img_list

def dogs_class_balanced_split(
        full_train_set, 
        train_size=0.9, 
        down_frac=0, # downsample fraction
        few_shot=False,
    ):
    imgs_by_class = defaultdict(list)
    for img, label in full_train_set:
        imgs_by_class[label].append(img)
    
    train_set, val_set = [], []

    for cls, imgs in imgs_by_class.items():
        random.shuffle(imgs)
        split_idx = int(len(imgs) * train_size)
        val_set.extend([(img, cls) for img in imgs[split_idx:]])
        train_imgs = [(img, cls) for img in imgs[:split_idx]]
        if down_frac > 0 or few_shot:
            k = int(len(imgs) * down_frac) if down_frac > 0 else few_shot
            train_imgs = random.sample(train_imgs, k=k)

        train_set.extend(train_imgs)

    random.shuffle(train_set)
    random.shuffle(val_set)
    return train_set, val_set

def load_stanford_dogs(
    dataset_configs,
    transform_train,
    transform_test,
    seed_worker_fn,
    seed_generator,
    logger,
    test_batch_size=None,
    debug=False
):
    '''
    Expects file structure like this:
        stanford_dogs/    
    ├── Images/
    │   └── <class_name>/
    │       └── <image_id>.jpg         
    ├── lists/
    │   ├── file_list.mat  
    │   ├── train_list.mat
    │   └── test_list.mat
    '''
    print('in')
    batch_size = dataset_configs["batch_size"]
    downsample_fraction = dataset_configs["downsample_fraction"]
    few_shot = dataset_configs["few_shot"]
    train_split = dataset_configs["train_split"]
    num_workers = dataset_configs["num_workers"]
    if not test_batch_size:
        test_batch_size = batch_size

    test_list_pt = "./data/stanford_dogs/lists/test_list.mat"
    train_list_pt = "./data/stanford_dogs/lists/train_list.mat"
    test_set = load_img_paths(test_list_pt)
    full_train_set = load_img_paths(train_list_pt)

    train_set, val_set = dogs_class_balanced_split(
        full_train_set, 
        train_size=train_split, 
        down_frac=downsample_fraction
    )
    if downsample_fraction > 0 or few_shot:
        logger.info(f"Downsampling by: {downsample_fraction}. Few shot: {few_shot}")

    if debug:
        train_set = train_set[:min(50, len(train_set))]
        val_set = val_set[:min(50, len(train_set))]
        test_set = test_set[:min(50, len(train_set))]
        batch_size=8

    train_dataset = StanfordDogsDataset(train_set, transform_train)
    val_dataset = StanfordDogsDataset(val_set, transform_test)
    test_dataset = StanfordDogsDataset(test_set, transform_test)

    if downsample_fraction > 0 and downsample_fraction < 0.2:
        logger.info(f"Downsample fraction is small - CHANGING TRAIN BATCH SIZE TO 32")
        train_batch_size = 32
    else:
        train_batch_size = batch_size

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers, worker_init_fn=seed_worker_fn, generator=seed_generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker_fn, generator=seed_generator)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker_fn, generator=seed_generator)
    return train_loader, val_loader, test_loader

class StanfordDogsDataset(Dataset):
    def __init__(self, img_paths, transform=None):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx][0]
        label = self.img_paths[idx][1]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)
            
        return img, label
