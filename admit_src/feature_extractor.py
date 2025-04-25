"""
Feature extraction module for AdMiT.
Uses a pre-trained model (e.g., DenseNet201) to extract features
for Kernel Mean Embedding (KME) calculations.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset # Use Dataset from torch.utils.data
import numpy as np
from tqdm import tqdm
import logging
import os

from . import config
from . import utils
from .data_loader import ImageDataset # Use the ImageDataset from our data_loader

class FeatureExtractor:
    """
    Extracts features using a specified pretrained model.
    Defaults to DenseNet201 
    """
    def __init__(self, model_name=config.KME_FEATURE_EXTRACTOR, device=config.DEVICE):
        """
        Initializes the feature extractor.
        Args:
            model_name (str): Name of the torchvision model to use (e.g., 'densenet201').
            device (str): Device to run the model on ('cuda' or 'cpu').
        """
        self.device = torch.device(device)
        self.model_name = model_name
        logging.info(f"Initializing Feature Extractor with model: {self.model_name} on device: {self.device}")

        # Load the specified pretrained model
        try:
            if model_name == 'densenet201':
                weights = models.DenseNet201_Weights.IMAGENET1K_V1
                self.model = models.densenet201(weights=weights)
                # Get the feature dimension before the classifier
                self.feature_dim = self.model.classifier.in_features
                # Remove the final classification layer
                self.feature_extractor_model = nn.Sequential(*list(self.model.children())[:-1])
                # Standard DenseNet preprocessing
                self.preprocess = weights.transforms()
                logging.info(f"Loaded pretrained {model_name}. Feature dimension: {self.feature_dim}")

            # Add other models if needed (e.g., ResNet, ViT)
            # elif model_name == 'resnet50':
            #     weights = models.ResNet50_Weights.IMAGENET1K_V2
            #     self.model = models.resnet50(weights=weights)
            #     self.feature_dim = self.model.fc.in_features
            #     self.feature_extractor_model = nn.Sequential(*list(self.model.children())[:-1])
            #     self.preprocess = weights.transforms()
            #     logging.info(f"Loaded pretrained {model_name}. Feature dimension: {self.feature_dim}")

            else:
                raise ValueError(f"Unsupported feature extractor model: {model_name}")

        except Exception as e:
            logging.error(f"Failed to load pretrained model {model_name}: {e}")
            raise

        self.feature_extractor_model.eval()  # Set to evaluation mode
        self.feature_extractor_model = self.feature_extractor_model.to(self.device)

        # Verify feature dimension matches config if specified (optional)
        if hasattr(config, 'KME_FEATURE_DIM') and config.KME_FEATURE_DIM != self.feature_dim:
            logging.warning(f"Configured KME_FEATURE_DIM ({config.KME_FEATURE_DIM}) does not match model's feature dim ({self.feature_dim}). Using model's dim.")


    def extract_features(self, data_source, batch_size=32):
        """
        Extracts features from images provided either as a numpy array or a DataLoader.
        Args:
            data_source (numpy.ndarray or torch.utils.data.DataLoader):
                         Input images (N, H, W, C or N, C, H, W) or a DataLoader.
            batch_size (int): Batch size to use if data_source is a numpy array.
        Returns:
            numpy.ndarray: Extracted features of shape (N, feature_dim).
        """
        if isinstance(data_source, np.ndarray):
            logging.info(f"Extracting features from numpy array of shape {data_source.shape}")
            # Create an ImageDataset using the provided numpy array
            # We assume labels are not needed for feature extraction here
            dataset = ImageDataset(images=data_source, labels=None, transform=self.preprocess)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=config.NUM_WORKERS)
        elif isinstance(data_source, DataLoader):
            logging.info("Extracting features using provided DataLoader.")
            dataloader = data_source
            # Assume the dataloader's dataset already has appropriate transforms applied.
            # If not, this might need adjustment or a check.
        else:
            raise TypeError("data_source must be either a numpy array or a PyTorch DataLoader.")

        all_features = []
        with torch.no_grad(): # Ensure no gradients are computed
            for batch_data in tqdm(dataloader, desc=f"Extracting features ({self.model_name})"):
                # Dataloader might return images and labels, we only need images
                if isinstance(batch_data, (list, tuple)):
                    batch_images = batch_data[0]
                else:
                    batch_images = batch_data

                batch_images = batch_images.to(self.device)

                # Extract features
                batch_features = self.feature_extractor_model(batch_images)

                # Apply global average pooling (common practice after feature extraction)
                # For models like DenseNet/ResNet, this pools the spatial dimensions
                # For ViT, the output might already be pooled or have a CLS token; adjust if needed.
                if len(batch_features.shape) == 4: # Typical output for CNN features (N, C, H, W)
                    batch_features = nn.functional.adaptive_avg_pool2d(batch_features, (1, 1))
                    batch_features = batch_features.view(batch_features.size(0), -1) # Flatten to (N, C)
                elif len(batch_features.shape) == 3: # Possible output for ViT (N, SeqLen, Dim) - take CLS token?
                     # Assuming CLS token is first: batch_features = batch_features[:, 0]
                     # Or average pool sequence dim: batch_features = batch_features.mean(dim=1)
                     # For now, assume typical CNN output handled above. Needs adjustment for ViT.
                     logging.warning(f"Feature extractor output shape {batch_features.shape} might need specific handling (e.g., for ViT). Assuming simple flatten.")
                     batch_features = batch_features.view(batch_features.size(0), -1) # Flatten

                all_features.append(batch_features.cpu().numpy())

        if not all_features:
            logging.warning("No features were extracted.")
            return np.array([]).reshape(0, self.feature_dim)

        features_np = np.vstack(all_features)
        logging.info(f"Extracted features shape: {features_np.shape}")
        return features_np


if __name__ == '__main__':
    # Example Usage: Extract features from dummy data
    print("--- Testing Feature Extractor ---")
    try:
        # Create dummy data (resembling CIFAR images)
        dummy_images = np.random.rand(10, 32, 32, 3).astype(np.float32) # 10 images, HWC format

        # Initialize extractor
        extractor = FeatureExtractor(model_name=config.KME_FEATURE_EXTRACTOR) # Use model from config

        # Extract features
        features = extractor.extract_features(dummy_images, batch_size=4)

        print(f"Successfully extracted features. Shape: {features.shape}")
        assert features.shape == (10, extractor.feature_dim)
        utils.logging.info("Feature extractor test successful.")

    except Exception as e:
        utils.logging.error(f"Feature extractor test failed: {e}", exc_info=True)