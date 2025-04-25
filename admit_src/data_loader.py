"""
Handles dataset downloading, loading, and preprocessing.
Includes specific functions for CIFAR-100-C based on the prototype script,
and more complete loaders for other datasets mentioned in the AdMiT paper.
"""

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from PIL import Image
import requests
import tarfile
import io
import logging
import pickle
import glob # For Digits-Five style loading
import random

from . import config
from . import utils

# --- CIFAR-100-C Specific Functions (from rkme_cifar100c_pipeline.py) ---

def download_cifar100c(data_dir=config.DATA_DIR):
    """
    Downloads and extracts the CIFAR-100-C dataset if not present.
    Reference: https://zenodo.org/record/3555552
    Args:
        data_dir (str): The base directory to download and extract data.
    Returns:
        str: Path to the extracted CIFAR-100-C directory.
    """
    cifar100c_path = os.path.join(data_dir, "CIFAR-100-C")
    cifar100c_tar_url = "https://zenodo.org/record/3555552/files/CIFAR-100-C.tar"

    if os.path.exists(cifar100c_path):
        logging.info(f"CIFAR-100-C already exists at {cifar100c_path}")
        expected_files = ['gaussian_noise.npy', 'labels.npy']
        if all(os.path.exists(os.path.join(cifar100c_path, f)) for f in expected_files):
             return cifar100c_path
        else:
             logging.warning(f"CIFAR-100-C directory found but seems incomplete. Consider removing '{cifar100c_path}' and re-running.")

    logging.info(f"Downloading CIFAR-100-C dataset from {cifar100c_tar_url}...")
    os.makedirs(data_dir, exist_ok=True)

    try:
        response = requests.get(cifar100c_tar_url, stream=True)
        response.raise_for_status()

        logging.info("Extracting tar file...")
        with io.BytesIO(response.content) as tar_stream:
            with tarfile.open(fileobj=tar_stream, mode='r') as tar_file:
                tar_file.extractall(path=data_dir)

        logging.info(f"Downloaded and extracted CIFAR-100-C to {cifar100c_path}")

        if not os.path.exists(cifar100c_path) or not os.path.exists(os.path.join(cifar100c_path, 'labels.npy')):
             raise FileNotFoundError(f"Extraction seems incomplete in {cifar100c_path}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading CIFAR-100-C: {e}")
        raise
    except tarfile.TarError as e:
        logging.error(f"Error extracting CIFAR-100-C tar file: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during CIFAR-100-C download/extraction: {e}")
        raise

    return cifar100c_path

def load_cifar100c_corruption(base_path, corruption_name, severity=1, num_images=None, indices=None):
    """Loads a specific corruption from the CIFAR-100-C dataset."""
    corruption_file = os.path.join(base_path, f"{corruption_name}.npy")
    labels_file = os.path.join(base_path, "labels.npy")

    if not os.path.exists(corruption_file) or not os.path.exists(labels_file):
        logging.error(f"Corruption file ({corruption_file}) or labels file ({labels_file}) not found.")
        return None, None

    try:
        all_corruption_images = np.load(corruption_file)
        all_labels = np.load(labels_file)

        images_per_severity = 10000
        total_expected_images = 5 * images_per_severity
        num_labels = 10000

        if all_labels.shape[0] != num_labels:
             logging.warning(f"Expected 10000 labels but found {all_labels.shape[0]}.")
             # Adjust label logic if necessary

        # Handle potential variations in .npy file structure
        if all_corruption_images.shape[0] == total_expected_images:
            start_idx = (severity - 1) * images_per_severity
            end_idx = severity * images_per_severity
            severity_images = all_corruption_images[start_idx:end_idx]
            # Labels correspond to the original test set, repeated implicitly for each severity
            severity_labels = all_labels[:images_per_severity]
        elif all_corruption_images.shape[0] == images_per_severity:
             # Assume file only contains data for one severity (or it's severity 1 implicitly)
             if severity != 1:
                 logging.warning(f"Corruption file {corruption_file} only contains {images_per_severity} images, but severity {severity} was requested. Using available images.")
             severity_images = all_corruption_images
             severity_labels = all_labels[:images_per_severity]
        else:
             logging.error(f"Unexpected number of images ({all_corruption_images.shape[0]}) in {corruption_file}. Expected {total_expected_images} or {images_per_severity}.")
             return None, None


        if severity_images.shape[0] != images_per_severity or severity_labels.shape[0] != images_per_severity:
            logging.error(f"Mismatch in expected image/label count for severity {severity}. Cannot proceed reliably.")
            return None, None

        if indices is None:
            if num_images is None or num_images >= images_per_severity:
                selected_indices = np.arange(images_per_severity)
            else:
                num_to_select = min(num_images, images_per_severity)
                selected_indices = np.random.choice(images_per_severity, num_to_select, replace=False)
        else:
            if np.any(indices >= images_per_severity) or np.any(indices < 0):
                 logging.error(f"Provided indices are out of bounds (max index {images_per_severity - 1}).")
                 return None, None
            selected_indices = indices

        selected_images = severity_images[selected_indices]
        selected_labels = severity_labels[selected_indices]
        selected_images = selected_images.astype(np.float32) / 255.0

        logging.info(f"Loaded {corruption_name} (severity {severity}): {len(selected_images)} images")
        return selected_images, selected_labels

    except Exception as e:
        logging.error(f"Error loading {corruption_name} (severity {severity}): {e}", exc_info=True)
        return None, None


# --- ImageNet-C Loading ---
# Assumes ImageNet-C is pre-generated and follows structure: corruption/severity/class/image.jpeg
def load_imagenet_c_corruption(base_path, corruption_name, severity=1, transform=None):
    """Loads ImageNet-C data for a specific corruption and severity using ImageFolder."""
    corruption_path = os.path.join(base_path, corruption_name, str(severity))
    if not os.path.isdir(corruption_path):
        logging.error(f"ImageNet-C path not found or not a directory: {corruption_path}")
        logging.error("Please ensure ImageNet-C is generated and placed correctly.")
        return None
    try:
        dataset = torchvision.datasets.ImageFolder(root=corruption_path, transform=transform)
        if len(dataset) == 0:
             logging.warning(f"No images found in ImageNet-C path: {corruption_path}")
             return None
        logging.info(f"Loaded ImageNet-C {corruption_name} (severity {severity}) with {len(dataset)} images.")
        return dataset
    except Exception as e:
        logging.error(f"Error loading ImageNet-C from {corruption_path}: {e}")
        return None

# --- ACDC Dataset Class ---
class ACDCDataset(Dataset):
    """Custom Dataset for ACDC (Semantic Segmentation)."""
    def __init__(self, root_dir, condition='fog', split='train', transform=None, target_transform=None):
        self.root_dir = root_dir
        self.condition = condition
        self.split = split # 'train', 'val', 'test', 'ref' etc. (ACDC structure might vary)
        self.transform = transform
        self.target_transform = target_transform # For mask

        self.img_dir = os.path.join(root_dir, 'rgb_anon', condition, split)
        self.mask_dir = os.path.join(root_dir, 'gt', condition, split)

        if not os.path.isdir(self.img_dir) or not os.path.isdir(self.mask_dir):
             raise FileNotFoundError(f"ACDC image ({self.img_dir}) or mask ({self.mask_dir}) directory not found for condition '{condition}', split '{split}'.")

        self.image_files = sorted(glob.glob(os.path.join(self.img_dir, '**', '*.png'), recursive=True))
        self.mask_files = sorted(glob.glob(os.path.join(self.mask_dir, '**', '*_labelIds.png'), recursive=True))

        if len(self.image_files) != len(self.mask_files) or len(self.image_files) == 0:
             logging.warning(f"Mismatch or zero files found for ACDC {condition}/{split}. Images: {len(self.image_files)}, Masks: {len(self.mask_files)}")
             # Fallback or error
             self.image_files = []
             self.mask_files = []


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path) # Masks are usually grayscale (label IDs)

            # Apply transformations - Need transforms that handle image and mask together
            # For simplicity, applying separately here. Adapt if using albumentations etc.
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)
            else: # Default mask conversion if no specific transform
                 mask = torch.from_numpy(np.array(mask, dtype=np.int64))


            # Map Cityscapes train IDs if needed (ACDC uses Cityscapes labels)
            # Implement mapping function here if necessary

            return image, mask

        except Exception as e:
            logging.error(f"Error loading ACDC sample {idx} (img: {img_path}): {e}")
            # Return dummy data or raise error
            dummy_size = (512, 1024) # Example size
            return torch.zeros(3, *dummy_size), torch.zeros(dummy_size, dtype=torch.long)


# --- Generic Image Dataset Class (from previous step) ---
class ImageDataset(Dataset):
    """Generic Dataset class for image data from numpy arrays."""
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform
        if self.labels is not None and len(self.images) != len(self.labels):
            raise ValueError("Number of images and labels must match.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] if self.labels is not None else -1
        if image.dtype == np.float32 or image.dtype == np.float64:
            if image.max() <= 1.0 and image.min() >= 0.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        if image.shape[0] == 1 and len(image.shape) == 3: # Grayscale C,H,W
             image = image.squeeze(0) # H,W
        if len(image.shape) == 2: # Grayscale H,W
             pil_image = Image.fromarray(image, mode='L').convert('RGB') # Convert to RGB
        elif image.shape[0] == 3 and len(image.shape) == 3: # NCHW format
             image = image.transpose(1, 2, 0) # Convert to HWC
             pil_image = Image.fromarray(image)
        elif len(image.shape) == 3 and image.shape[-1] == 1: # HWC with 1 channel
             pil_image = Image.fromarray(image.squeeze(-1), mode='L').convert('RGB')
        elif len(image.shape) == 3 and image.shape[-1] == 3: # HWC
            pil_image = Image.fromarray(image)
        else:
            logging.error(f"Unexpected image shape {image.shape} at index {idx}.")
            return torch.zeros(3, 32, 32), torch.tensor(-1, dtype=torch.long) # Dummy data

        if self.transform:
            pil_image = self.transform(pil_image)

        return pil_image, torch.tensor(label, dtype=torch.long)

# --- Transformations ---
def get_transforms(dataset_name, split='train', img_size=224):
    """Gets data transformations."""
    # Use ImageNet stats as default, adjust if needed
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    interpolation = transforms.InterpolationMode.BICUBIC # Common for ViT

    if split == 'train':
        # Basic augmentation for source training
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=interpolation),
            transforms.CenterCrop(img_size), # Use CenterCrop after resize
            # transforms.RandomResizedCrop(img_size, interpolation=interpolation), # Alternative
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else: # Validation/Test/Adaptation
        # No augmentation, just resize and normalize
        return transforms.Compose([
            transforms.Resize(img_size, interpolation=interpolation),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

def get_segmentation_transforms(split='train', img_size=(512, 1024)):
     """Gets transformations specifically for segmentation tasks (image + mask)."""
     # This requires transforms that operate on both image and mask simultaneously.
     # Using basic separate transforms here. Consider libraries like albumentations.
     mean = [0.485, 0.456, 0.406]
     std = [0.229, 0.224, 0.225]

     img_transform = transforms.Compose([
         transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BILINEAR), # Use BILINEAR for images
         transforms.ToTensor(),
         transforms.Normalize(mean=mean, std=std)
     ])
     # Mask transform: resize using NEAREST, convert to LongTensor
     mask_transform = transforms.Compose([
         transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
         transforms.Lambda(lambda x: torch.from_numpy(np.array(x, dtype=np.int64))) # Convert PIL mask to LongTensor
     ])
     # TODO: Implement augmentations (like random crop, flip) that apply to both
     return img_transform, mask_transform


# --- Data Loading ---
def get_dataloader(dataset_name, split, batch_size, transform=None, target_transform=None, num_workers=4, shuffle=None, num_images=None, indices=None, **kwargs):
    """
    Main function to get a DataLoader for a specified dataset and split/domain.
    Args:
        dataset_name (str): Name of the dataset (e.g., 'cifar100c', 'digits_five', 'cityscapes').
        split (str): 'train', 'val', 'test', or specific domain/corruption identifier.
        batch_size (int): Batch size.
        transform (callable, optional): Image transform. Auto-selected if None.
        target_transform (callable, optional): Target/Mask transform (for segmentation).
        num_workers (int): DataLoader workers.
        shuffle (bool, optional): Shuffle data? Defaults based on split.
        num_images (int, optional): Max images for CIFAR-100-C loading.
        indices (np.array, optional): Specific indices for CIFAR-100-C loading.
        **kwargs: Additional args passed based on dataset type
                  (e.g., corruption_name, severity for C-datasets).
    Returns:
        torch.utils.data.DataLoader: The DataLoader instance.
    """
    if shuffle is None:
        shuffle = (split == 'train')

    dataset = None
    is_segmentation = dataset_name in ['cityscapes', 'acdc']

    # Select default transforms if not provided
    if transform is None and not is_segmentation:
         img_size = 224 # Default size for ViT/DenseNet
         transform = get_transforms(dataset_name, split, img_size=img_size)
    elif transform is None and is_segmentation:
         # Get default segmentation transforms (may need size adjustment)
         transform, target_transform = get_segmentation_transforms(split)


    dataset_path = config.DATASETS.get(dataset_name, config.DATA_DIR)
    logging.info(f"Attempting to load dataset: {dataset_name}, split/domain: {split}, path: {dataset_path}")

    # --- CIFAR-100-C ---
    if dataset_name == 'cifar100c':
        corruption_name = kwargs.get('corruption_name')
        severity = kwargs.get('severity')
        if corruption_name is None or severity is None:
            raise ValueError("corruption_name and severity must be provided for cifar100c")
        cifar100c_path = download_cifar100c(config.DATA_DIR)
        images, labels = load_cifar100c_corruption(
            cifar100c_path, corruption_name, severity, num_images=num_images, indices=indices
        )
        if images is None: raise RuntimeError("Failed to load CIFAR-100-C data.")
        dataset = ImageDataset(images, labels, transform=transform)

    # --- ImageNet-C ---
    elif dataset_name == 'imagenet_c':
        corruption_name = kwargs.get('corruption_name')
        severity = kwargs.get('severity')
        if corruption_name is None or severity is None:
            raise ValueError("corruption_name and severity must be provided for imagenet_c")
        if not isinstance(dataset_path, str): raise ValueError("Invalid path config for ImageNet-C")
        dataset = load_imagenet_c_corruption(dataset_path, corruption_name, severity, transform=transform)
        if dataset is None: raise RuntimeError("Failed to load ImageNet-C data.")

    # --- Digits-Five ---
    # 'split' argument here refers to the domain name (mnist, svhn, etc.)
    elif dataset_name == 'digits_five':
        domain_name = split
        if not isinstance(dataset_path, dict) or domain_name not in dataset_path:
             raise ValueError(f"Domain '{domain_name}' not configured in config.DATASETS['digits_five']")
        domain_path = dataset_path[domain_name]
        if not os.path.isdir(domain_path):
             raise FileNotFoundError(f"Digits-Five domain path not found: {domain_path}.")

        # Handle torchvision datasets vs ImageFolder
        tv_map = {'mnist': torchvision.datasets.MNIST,
                  'svhn': torchvision.datasets.SVHN,
                  'usps': torchvision.datasets.USPS}
        if domain_name in tv_map:
            # SVHN uses 'train'/'test' split arg, MNIST/USPS use 'train' boolean
            split_arg = 'train' if domain_name == 'svhn' and shuffle else 'test' if domain_name == 'svhn' else shuffle
            try:
                dataset = tv_map[domain_name](root=os.path.dirname(domain_path), # Root is parent dir for torchvision
                                               split=split_arg if domain_name in ['svhn','usps'] else None, # split arg for SVHN/USPS
                                               train=shuffle if domain_name == 'mnist' else None, # train arg for MNIST
                                               download=True, transform=transform)
                logging.info(f"Loaded Digits-Five domain '{domain_name}' using torchvision.")
            except Exception as e:
                 logging.error(f"Failed to load torchvision dataset {domain_name}: {e}. Check structure/args.")
                 raise
        elif domain_name in ['mnistm', 'syndigits']:
            # Assume ImageFolder structure for MNIST-M, SynDigits
             dataset = torchvision.datasets.ImageFolder(root=domain_path, transform=transform)
             logging.info(f"Loaded Digits-Five domain '{domain_name}' using ImageFolder.")
        else:
            raise ValueError(f"Unknown Digits-Five domain: {domain_name}")


    # --- Cityscapes ---
    elif dataset_name == 'cityscapes':
         if not isinstance(dataset_path, str): raise ValueError("Invalid path config for Cityscapes")
         try:
             # Cityscapes loader requires specific mode and target_type
             dataset = torchvision.datasets.Cityscapes(
                 root=dataset_path,
                 split=split, # Typically 'train', 'val', 'test'
                 mode='fine', # 'fine' for semantic segmentation
                 target_type='semantic', # Get semantic segmentation masks
                 transform=transform,
                 target_transform=target_transform
             )
             logging.info(f"Loaded Cityscapes split '{split}'.")
         except Exception as e:
              logging.error(f"Failed to load Cityscapes dataset from {dataset_path}: {e}")
              raise

    # --- ACDC ---
    elif dataset_name == 'acdc':
         if not isinstance(dataset_path, str): raise ValueError("Invalid path config for ACDC")
         condition = kwargs.get('condition', 'fog') # Default condition if not specified
         try:
             dataset = ACDCDataset(
                 root_dir=dataset_path,
                 condition=condition,
                 split=split, # ACDC might use 'train', 'val', 'test' or others
                 transform=transform,
                 target_transform=target_transform
             )
             logging.info(f"Loaded ACDC condition '{condition}', split '{split}'.")
         except Exception as e:
              logging.error(f"Failed to load ACDC dataset: {e}")
              raise

    # --- Standard Torchvision Datasets (Example: CIFAR10) ---
    elif dataset_name == 'cifar10':
        if not isinstance(dataset_path, str): dataset_path=config.DATA_DIR # Default save loc
        is_train = (split == 'train')
        dataset = torchvision.datasets.CIFAR10(root=dataset_path, train=is_train, download=True, transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if dataset is None:
         raise RuntimeError(f"Dataset object was not created for {dataset_name}/{split}.")

    # Handle loading 'all' data for KME feature extraction if not handled above
    if split == 'all' and isinstance(dataset, (torchvision.datasets.VisionDataset)):
        # For datasets where we loaded specific splits ('train'/'test'),
        # 'all' might mean combining them or loading the whole set if possible.
        # This part needs careful implementation based on how each dataset is structured.
        # For simplicity, we currently handle 'all' via specific loader args (e.g., num_images=None for CIFAR-C)
        # or by assuming the standard loader loads everything (less common).
        logging.warning(f"Split 'all' requested for {dataset_name}. Ensure the loader handles loading all relevant data.")


    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if config.DEVICE == 'cuda' else False,
        drop_last=(split == 'train')
    )

    logging.info(f"Created DataLoader for {dataset_name} ({split}). Samples: {len(dataset)}, Batches: {len(dataloader)}, BS: {batch_size}, Shuffle: {shuffle}")
    return dataloader


if __name__ == '__main__':
    # Example Usage:
    print("--- Testing Data Loaders ---")
    utils.setup_logging()
    config.DEVICE = 'cpu' # Force CPU for testing if GPU not needed/available

    # Test CIFAR-100-C
    try:
        print("\nTesting CIFAR-100-C...")
        loader = get_dataloader( 'cifar100c', 'test', batch_size=4, corruption_name='fog', severity=1, num_images=10)
        img, lbl = next(iter(loader))
        print(f"  CIFAR-100-C batch shape: {img.shape}, labels: {lbl.shape}")
    except Exception as e: print(f"  CIFAR-100-C failed: {e}")

    # Test Digits-Five (MNIST)
    try:
        print("\nTesting Digits-Five (MNIST)...")
        # Need MNIST data available or torchvision will download it
        mnist_path = config.DATASETS.get('digits_five', {}).get('mnist')
        if not mnist_path: mnist_path = os.path.join(config.DATA_DIR, 'digits_five', 'mnist') # Default guess
        os.makedirs(os.path.dirname(mnist_path), exist_ok=True) # Ensure parent exists for download
        loader = get_dataloader('digits_five', split='mnist', batch_size=4, shuffle=False)
        img, lbl = next(iter(loader))
        print(f"  Digits-Five (MNIST) batch shape: {img.shape}, labels: {lbl.shape}")
    except Exception as e: print(f"  Digits-Five (MNIST) failed: {e}")

    # Test Cityscapes (will fail if dataset not downloaded)
    # try:
    #     print("\nTesting Cityscapes...")
    #     loader = get_dataloader('cityscapes', split='val', batch_size=2)
    #     img, mask = next(iter(loader))
    #     print(f"  Cityscapes batch shape: Image {img.shape}, Mask {mask.shape}")
    # except Exception as e: print(f"  Cityscapes failed (likely needs download): {e}")

    print("\n--- Data Loader testing finished ---")