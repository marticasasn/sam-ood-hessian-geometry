import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

# CIFAR-10 Normalization Statistics
# These values are standard for CIFAR-10 and essential for ResNet performance.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

class CIFAR10CDataset(Dataset):
    """
    Custom Dataset for CIFAR-10-C (Corrupted).
    Each corruption file (e.g., 'snow.npy') has 50,000 images (10k per severity level, 1-5).
    """
    def __init__(self, root, corruption, severity, transform=None):
        """
        Args:
            root (string): Path to the CIFAR-10-C folder in Kaggle input.
            corruption (string): Name of corruption (e.g., 'shot_noise').
            severity (int): Severity level 1 (lowest) to 5 (highest).
            transform (callable, optional): Standard torchvision transforms.
        """
        assert 1 <= severity <= 5, "Severity must be between 1 and 5."
        self.root = root
        self.transform = transform
        
        # Define paths for the specific corruption and the global labels file
        data_path = os.path.join(root, f"{corruption}.npy")
        labels_path = os.path.join(root, "labels.npy")
        
        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(f"Check your Kaggle input path. Files not found in {root}")

        # Loading the full .npy into memory. 
        # On Kaggle, this is efficient as the files are ~150MB each.
        all_images = np.load(data_path)
        all_labels = np.load(labels_path)
        
        # Each severity level takes 10,000 images in sequence.
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        self.images = all_images[start_idx:end_idx]
        self.labels = all_labels[start_idx:end_idx]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Image is (32, 32, 3) uint8 NumPy array
        img = self.images[idx]
        label = self.labels[idx]
        
        # Must convert to PIL Image to apply standard torchvision transforms
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
            
        return img, int(label)

def get_train_loader(batch_size=128):
    """
    Standard CIFAR-10 Train Loader.
    Uses RandomCrop (with padding=4) and HorizontalFlip.
    SAM often relies on these to find more robust, flatter minima.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    # download=True is safe on Kaggle; it will verify or download to /kaggle/working/data
    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)

    return DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True)

def get_test_loader(batch_size=100):
    """
    Standard CIFAR-10 Test Loader (In-Distribution Evaluation).
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return DataLoader(testset, batch_size=batch_size, shuffle=False, pin_memory=True)

CIFAR10C_CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]

def get_cifar10c_loader(corruption, severity, root_dir='/kaggle/input/cifar10c', batch_size=100):
    """
    CIFAR-10-C Loader (Out-of-Distribution Evaluation).
    """
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    dataset = CIFAR10CDataset(root=root_dir, corruption=corruption, severity=severity, transform=transform_test)

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
