# test_env.py
import torch
import torchvision
import torchinfo
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

print("All packages imported successfully!")

if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is NOT available.")
