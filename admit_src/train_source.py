"""
Script to pre-train Parameter-Efficient Tuning (PET) modules on source domains
and generate their corresponding Kernel Mean Embedding (KME) representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR # Example scheduler
from torch.utils.data import DataLoader
import os
import argparse
import logging
import numpy as np
from tqdm import tqdm
import time
import copy # For deep copying model/config if needed

# Import local modules
from . import config
from . import utils
from . import data_loader
from . import feature_extractor
from . import kme
from . import pet_modules
# Need to import specific PET implementations based on config
# from .pet_modules.lora import LoRA
# from .pet_modules.adapter import Adapter
# from .pet_modules.vpt import VPT

def train_one_epoch(model, dataloader, criterion, optimizer, device, scheduler=None):
    """Trains the model for one epoch."""
    model.train() # Set model to training mode (important for PET modules)
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for batch in progress_bar:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs) # Forward pass through PET module + base model
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if scheduler is not None and isinstance(scheduler, CosineAnnealingLR): # Step per batch for cosine
             scheduler.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_one_epoch(model, dataloader, criterion, device):
    """Evaluates the model on the validation set for one epoch."""
    model.eval() # Set model to evaluation mode
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation", leave=False)
        for batch in progress_bar:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())

    epoch_loss = total_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def train_source_module(
    source_dataset_name,
    source_domain_id, # e.g., corruption name, domain name like 'mnist'
    pet_method_name,
    pet_config_dict,
    base_model_name,
    num_classes,
    output_pet_path,
    output_kme_path,
    device
    ):
    """
    Trains a single PET module for a specific source domain and saves it,
    then generates and saves its KME representation.
    """
    logging.info(f"--- Training Source Module for Domain: {source_domain_id} ---")

    # 1. Load Data
    logging.info("Loading data...")
    try:
        # Use source_domain_id as the 'split' for datasets like Digits-Five
        # For C-datasets, source_domain_id would be the corruption name/severity
        if source_dataset_name == 'cifar100c':
            corruption_name, severity_str = source_domain_id.split('_sev') # Assumes format like 'gaussian_noise_sev1'
            severity = int(severity_str)
            train_loader = data_loader.get_dataloader(
                source_dataset_name, split='train', batch_size=config.SOURCE_BATCH_SIZE,
                corruption_name=corruption_name, severity=severity
            )
            # Use a fixed validation set? Or another corruption/severity? Paper doesn't specify source val set.
            # Using same data for val here, but ideally use a separate set.
            val_loader = data_loader.get_dataloader(
                source_dataset_name, split='val', batch_size=config.SOURCE_BATCH_SIZE,
                corruption_name=corruption_name, severity=severity
            )
            # For KME features, load all data for this source domain without shuffling
            kme_data_loader = data_loader.get_dataloader(
                 source_dataset_name, split='all', batch_size=config.TARGET_BATCH_SIZE, # Larger batch for feature extraction
                 corruption_name=corruption_name, severity=severity, shuffle=False
            )

        elif source_dataset_name == 'digits_five':
            train_loader = data_loader.get_dataloader(
                source_dataset_name, split=source_domain_id, batch_size=config.SOURCE_BATCH_SIZE, shuffle=True
            )
            # Need a validation strategy for digits - maybe hold out some data?
            # Simple approach: use the same data for validation here.
            val_loader = data_loader.get_dataloader(
                source_dataset_name, split=source_domain_id, batch_size=config.SOURCE_BATCH_SIZE, shuffle=False
            )
            kme_data_loader = data_loader.get_dataloader(
                 source_dataset_name, split=source_domain_id, batch_size=config.TARGET_BATCH_SIZE, shuffle=False
            )
        else:
            # Add loading logic for other datasets (ImageNet-C, etc.)
            raise NotImplementedError(f"Data loading for dataset '{source_dataset_name}' not implemented.")

    except Exception as e:
        logging.error(f"Failed to load data for {source_domain_id}: {e}")
        return False

    # 2. Load Base Model
    logging.info(f"Loading base model: {base_model_name}")
    try:
        # Dynamically load model based on name
        if 'vit' in base_model_name:
            weights_enum = getattr(models, f"{base_model_name.upper()}_Weights", None) # Find weights enum
            weights = weights_enum.IMAGENET1K_V1 if weights_enum else None
            base_model = getattr(models, base_model_name)(weights=weights)
            # Replace classifier head
            in_features = base_model.heads.head.in_features
            base_model.heads.head = nn.Linear(in_features, num_classes)
        elif 'densenet' in base_model_name:
             weights_enum = getattr(models, f"{base_model_name.capitalize()}_Weights", None)
             weights = weights_enum.IMAGENET1K_V1 if weights_enum else None
             base_model = getattr(models, base_model_name)(weights=weights)
             in_features = base_model.classifier.in_features
             base_model.classifier = nn.Linear(in_features, num_classes)
        # Add other model families if needed
        else:
             raise ValueError(f"Unsupported base model type: {base_model_name}")

        base_model = base_model.to(device)
    except Exception as e:
        logging.error(f"Failed to load base model {base_model_name}: {e}")
        return False

    # Freeze base model parameters
    for param in base_model.parameters():
        param.requires_grad = False
    logging.info("Base model loaded and frozen.")

    # 3. Instantiate PET Module
    logging.info(f"Instantiating PET module: {pet_method_name}")
    try:
        # This assumes PET module constructor handles model modification
        # Or we need a separate apply_to_model step
        pet_class = getattr(pet_modules, pet_method_name.upper(), None) # e.g., pet_modules.LoRA
        if pet_class is None: # Try direct import if uppercase failed
             if pet_method_name == 'lora': from .pet_modules.lora import LoRA; pet_class = LoRA
             elif pet_method_name == 'adapter': from .pet_modules.adapter import Adapter; pet_class = Adapter
             elif pet_method_name == 'vpt': from .pet_modules.vpt import VPT; pet_class = VPT
             else: raise ValueError(f"PET Class for {pet_method_name} not found.")

        # Important: Pass a copy of the base model if PET modifies it in place?
        # Or ensure PET module can be attached without modifying original base_model structure
        # Let's assume PET module works with the passed base_model reference for now.
        pet_module = pet_class(base_model, pet_config_dict)
        pet_module = pet_module.to(device) # Ensure PET params are on device

        # Get only PET parameters for the optimizer
        pet_params_list = list(pet_module.get_pet_parameters())
        if not pet_params_list:
            logging.error("No trainable PET parameters found!")
            return False
        pet_module.print_trainable_parameters()

    except NotImplementedError:
        logging.error(f"PET method '{pet_method_name}' is not fully implemented.")
        return False
    except Exception as e:
        logging.error(f"Failed to instantiate PET module {pet_method_name}: {e}")
        return False

    # 4. Define Optimizer and Loss
    criterion = nn.CrossEntropyLoss()
    # Use optimizer settings from config
    if config.SOURCE_OPTIMIZER.lower() == 'adamw':
        optimizer = optim.AdamW(pet_params_list, lr=config.SOURCE_LEARNING_RATE, weight_decay=config.SOURCE_WEIGHT_DECAY)
    elif config.SOURCE_OPTIMIZER.lower() == 'sgd':
        optimizer = optim.SGD(pet_params_list, lr=config.SOURCE_LEARNING_RATE, momentum=0.9, weight_decay=config.SOURCE_WEIGHT_DECAY) # Example momentum
    else:
        logging.warning(f"Unsupported optimizer: {config.SOURCE_OPTIMIZER}. Using AdamW.")
        optimizer = optim.AdamW(pet_params_list, lr=config.SOURCE_LEARNING_RATE, weight_decay=config.SOURCE_WEIGHT_DECAY)

    scheduler = None
    if config.SOURCE_LR_SCHEDULE == 'cosine':
         scheduler = CosineAnnealingLR(optimizer, T_max=len(train_loader) * config.SOURCE_TRAIN_EPOCHS)
         logging.info("Using Cosine Annealing LR scheduler.")

    # 5. Training Loop
    logging.info(f"Starting training for {config.SOURCE_TRAIN_EPOCHS} epochs...")
    best_val_acc = 0.0
    best_pet_state = None

    for epoch in range(config.SOURCE_TRAIN_EPOCHS):
        start_time = time.time()
        train_loss, train_acc = train_one_epoch(pet_module, train_loader, criterion, optimizer, device, scheduler)
        val_loss, val_acc = evaluate_one_epoch(pet_module, val_loader, criterion, device)
        end_time = time.time()

        logging.info(f"Epoch {epoch+1}/{config.SOURCE_TRAIN_EPOCHS} | Time: {end_time-start_time:.2f}s | "
                     f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                     f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_pet_state = copy.deepcopy(pet_module.pet_state_dict()) # Save the best state
            logging.info(f"  New best validation accuracy: {best_val_acc:.4f}. Saving model state.")
            # Optional: Save immediately
            # torch.save(best_pet_state, output_pet_path)

    # 6. Save Final PET Module State
    if best_pet_state is None:
         logging.warning("No best state found (perhaps validation accuracy didn't improve). Saving final state.")
         best_pet_state = pet_module.pet_state_dict()

    try:
        os.makedirs(os.path.dirname(output_pet_path), exist_ok=True)
        torch.save(best_pet_state, output_pet_path)
        logging.info(f"Saved best PET module state to {output_pet_path}")
    except Exception as e:
        logging.error(f"Failed to save PET module state to {output_pet_path}: {e}")
        return False # Indicate failure

    # 7. Generate and Save KME Representation
    logging.info("Generating KME representation for the source domain...")
    try:
        # Load the feature extractor model for KME
        kme_extractor = feature_extractor.FeatureExtractor(model_name=config.KME_FEATURE_EXTRACTOR, device=device)

        # Extract features for the whole source dataset (using kme_data_loader)
        all_source_features = []
        logging.info("Extracting features for KME...")
        # We need to iterate through the loader to get all data if it wasn't loaded into memory
        # Re-create the dataloader to ensure we get all samples if needed
        kme_feature_loader = data_loader.get_dataloader(
                source_dataset_name, split='all', batch_size=config.TARGET_BATCH_SIZE, # Use target batch size
                corruption_name=corruption_name if source_dataset_name=='cifar100c' else None,
                severity=severity if source_dataset_name=='cifar100c' else None,
                shuffle=False, # IMPORTANT: No shuffle for KME consistency
                transform=kme_extractor.preprocess # Use KME extractor's preprocess
            )

        features_np = kme_extractor.extract_features(kme_feature_loader) # Pass loader directly

        if features_np is None or features_np.shape[0] == 0:
             raise ValueError("Feature extraction for KME returned no features.")

        # Compute KME representation
        kme_repr = kme.compute_kme_representation(features_np)
        if kme_repr is None:
            raise RuntimeError("Failed to compute KME representation.")

        # Save KME representation
        os.makedirs(os.path.dirname(output_kme_path), exist_ok=True)
        utils.save_kme(kme_repr, output_kme_path)
        logging.info(f"Saved KME representation to {output_kme_path}")

    except Exception as e:
        logging.error(f"Failed to generate or save KME representation: {e}", exc_info=True)
        # Decide if failure to generate KME should mark the whole process as failed
        return False # Indicate failure

    logging.info(f"--- Successfully Completed Training for Domain: {source_domain_id} ---")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train AdMiT Source PET Modules and Generate KMEs")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the source dataset (e.g., 'cifar100c', 'digits_five')")
    parser.add_argument('--domains', type=str, nargs='+', required=True, help="List of source domain IDs within the dataset (e.g., 'gaussian_noise_sev1' 'fog_sev1' or 'mnist' 'svhn')")
    parser.add_argument('--pet_method', type=str, default=config.PET_METHOD, choices=['lora', 'adapter', 'vpt'], help="PET method to use")
    # Add other relevant arguments to override config, e.g., epochs, lr, base_model
    parser.add_argument('--base_model', type=str, default=config.BASE_MODEL, help="Base model architecture")
    parser.add_argument('--epochs', type=int, default=config.SOURCE_TRAIN_EPOCHS, help="Number of training epochs")
    parser.add_argument('--lr', type=float, default=config.SOURCE_LEARNING_RATE, help="Learning rate")
    parser.add_argument('--output_dir', type=str, default=config.OUTPUT_DIR, help="Base directory for saving outputs")
    parser.add_argument('--device', type=str, default=config.DEVICE, help="Device ('cuda' or 'cpu')")

    args = parser.parse_args()

    # --- Override config ---
    config.PET_METHOD = args.pet_method
    config.BASE_MODEL = args.base_model
    config.SOURCE_TRAIN_EPOCHS = args.epochs
    config.SOURCE_LEARNING_RATE = args.lr
    config.DEVICE = args.device
    # Update output paths based on args.output_dir
    config.OUTPUT_DIR = args.output_dir
    config.PET_MODULE_DIR = os.path.join(args.output_dir, "pet_modules")
    config.KME_DIR = os.path.join(args.output_dir, "kmes")
    os.makedirs(config.PET_MODULE_DIR, exist_ok=True)
    os.makedirs(config.KME_DIR, exist_ok=True)

    utils.setup_logging() # Ensure logging is set up
    utils.set_seed(config.SEED)

    logging.info(f"Starting source training script with args: {args}")

    # Get number of classes based on dataset
    try:
        num_classes = config.NUM_CLASSES[args.dataset]
    except KeyError:
        logging.error(f"Number of classes not defined in config for dataset: {args.dataset}")
        exit(1)

    # Get PET config based on method
    try:
        pet_config_dict = config.PET_CONFIG[args.pet_method]
    except KeyError:
        logging.error(f"Configuration not defined in config for PET method: {args.pet_method}")
        exit(1)

    # Loop through specified source domains
    successful_domains = 0
    for domain_id in args.domains:
        logging.info(f"Processing source domain: {domain_id}")
        # Define output paths
        output_pet_filename = f"{args.dataset}_{domain_id}_{args.pet_method}.pt"
        output_kme_filename = f"{args.dataset}_{domain_id}_kme.pt"
        output_pet_filepath = os.path.join(config.PET_MODULE_DIR, output_pet_filename)
        output_kme_filepath = os.path.join(config.KME_DIR, output_kme_filename)

        success = train_source_module(
            source_dataset_name=args.dataset,
            source_domain_id=domain_id,
            pet_method_name=args.pet_method,
            pet_config_dict=pet_config_dict,
            base_model_name=args.base_model,
            num_classes=num_classes,
            output_pet_path=output_pet_filepath,
            output_kme_path=output_kme_filepath,
            device=args.device
        )
        if success:
            successful_domains += 1

    logging.info(f"Finished processing all domains. Successfully trained {successful_domains}/{len(args.domains)} modules.")