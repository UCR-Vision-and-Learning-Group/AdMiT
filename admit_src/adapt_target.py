"""
Script to perform test-time adaptation on target domains using AdMiT.
Loads pre-trained source PET modules and KMEs, processes target data
batch-by-batch, performs KME matching, module integration, optional tuning,
and evaluates performance.
"""

import torch
import torch.nn as nn
import os
import argparse
import logging
import numpy as np
import glob # For finding source files
from tqdm import tqdm
import time

# Import local modules
from . import config
from . import utils
from . import data_loader
from . import feature_extractor
from . import kme
from . import pet_modules # Needed to access PET classes for AdMiTModel
from . import admit_model

@torch.no_grad() # Ensure no gradients are calculated during evaluation/adaptation inference
def adapt_on_target(
    target_dataset_name,
    target_domain_id, # e.g., corruption name, domain name like 'usps'
    pet_method_name,
    pet_config_dict,
    base_model_name,
    num_classes,
    source_module_dir,
    source_kme_dir,
    device
    ):
    """
    Performs adaptation using AdMiT for a specific target domain.
    Args:
        target_dataset_name (str): Name of the target dataset (e.g., 'cifar100c').
        target_domain_id (str): Identifier for the target domain/split/corruption.
        pet_method_name (str): Name of the PET method used ('lora', 'adapter', 'vpt').
        pet_config_dict (dict): Configuration for the PET method.
        base_model_name (str): Name of the base model architecture.
        num_classes (int): Number of output classes.
        source_module_dir (str): Directory containing saved source PET module state dicts.
        source_kme_dir (str): Directory containing saved source KME representations.
        device (str): Device to run on ('cuda' or 'cpu').
    Returns:
        float: Accuracy on the target domain after adaptation. Returns -1.0 on failure.
    """
    logging.info(f"--- Adapting on Target Domain: {target_domain_id} ---")

    # 1. Load Target Data
    logging.info("Loading target data...")
    try:
        # Determine how to load target data based on dataset name and domain id
        # For C-datasets, target_domain_id specifies corruption/severity
        if target_dataset_name == 'cifar100c':
            corruption_name, severity_str = target_domain_id.split('_sev')
            severity = int(severity_str)
            target_loader = data_loader.get_dataloader(
                target_dataset_name, split='test', batch_size=config.TARGET_BATCH_SIZE,
                corruption_name=corruption_name, severity=severity, shuffle=False # No shuffle for TTA
            )
            # KME feature loader needs specific transforms and no shuffle
            kme_transform = feature_extractor.FeatureExtractor(model_name=config.KME_FEATURE_EXTRACTOR).preprocess
            kme_feature_loader = data_loader.get_dataloader(
                target_dataset_name, split='test_kme', batch_size=config.TARGET_BATCH_SIZE, # Different split name if needed
                corruption_name=corruption_name, severity=severity, shuffle=False,
                transform=kme_transform # Use KME extractor's transform
            )

        elif target_dataset_name == 'digits_five':
            target_loader = data_loader.get_dataloader(
                target_dataset_name, split=target_domain_id, batch_size=config.TARGET_BATCH_SIZE, shuffle=False
            )
            kme_transform = feature_extractor.FeatureExtractor(model_name=config.KME_FEATURE_EXTRACTOR).preprocess
            kme_feature_loader = data_loader.get_dataloader(
                target_dataset_name, split=target_domain_id, batch_size=config.TARGET_BATCH_SIZE, shuffle=False,
                transform=kme_transform
            )
        else:
            # Add loading logic for other datasets (ImageNet-C, ACDC, etc.)
            raise NotImplementedError(f"Target data loading for dataset '{target_dataset_name}' not implemented.")

    except Exception as e:
        logging.error(f"Failed to load target data for {target_domain_id}: {e}")
        return -1.0

    # 2. Load Base Model Architecture (weights might not matter if PET replaces output head)
    logging.info(f"Loading base model architecture: {base_model_name}")
    try:
         # Load architecture without pre-trained weights if PET handles the task head
        if 'vit' in base_model_name:
            base_model = getattr(models, base_model_name)(weights=None, num_classes=num_classes)
        elif 'densenet' in base_model_name:
             base_model = getattr(models, base_model_name)(weights=None, num_classes=num_classes)
        else:
             raise ValueError(f"Unsupported base model type: {base_model_name}")
        base_model = base_model.to(device)
    except Exception as e:
        logging.error(f"Failed to load base model architecture {base_model_name}: {e}")
        return -1.0

    # 3. Identify Source Modules and KMEs
    # Assume files are named like: {dataset}_{domain_id}_{pet_method}.pt and {dataset}_{domain_id}_kme.pt
    # We need to find all files matching the pattern *except* the target domain itself (if present in source dirs)
    logging.info(f"Identifying source modules from: {source_module_dir}")
    logging.info(f"Identifying source KMEs from: {source_kme_dir}")

    source_pet_paths = []
    source_kme_paths = []
    # Example pattern - adjust based on actual naming convention from train_source.py
    pet_pattern = os.path.join(source_module_dir, f"*_{pet_method_name}.pt")
    kme_pattern = os.path.join(source_kme_dir, "*_kme.pt")

    all_pet_files = glob.glob(pet_pattern)
    all_kme_files = glob.glob(kme_pattern)

    # Match PET files with KME files based on the domain identifier part of the filename
    kme_basenames = {os.path.basename(p).replace('_kme.pt', ''): p for p in all_kme_files}

    for pet_path in all_pet_files:
        pet_basename = os.path.basename(pet_path).replace(f'_{pet_method_name}.pt', '')
        # Avoid using the target domain as a source if it exists
        # This requires parsing domain_id from pet_basename accurately
        # Simple check: if basename itself is the target_domain_id string
        is_target_domain = (pet_basename.endswith(target_domain_id)) # Simple check, might need refinement

        if not is_target_domain and pet_basename in kme_basenames:
            source_pet_paths.append(pet_path)
            source_kme_paths.append(kme_basenames[pet_basename])
        elif pet_basename not in kme_basenames:
             logging.warning(f"Found PET module {pet_path} but no matching KME file. Skipping.")

    if not source_pet_paths or not source_kme_paths:
         logging.error(f"Could not find matching source PET modules and KME files in specified directories for method {pet_method_name}.")
         return -1.0

    logging.info(f"Found {len(source_pet_paths)} source modules/KMEs to use for target {target_domain_id}.")

    # 4. Instantiate AdMiTModel
    logging.info("Instantiating AdMiT model...")
    try:
        admit_instance = admit_model.AdMiTModel(
            base_model=base_model, # Pass the architecture
            num_classes=num_classes,
            pet_method_name=pet_method_name,
            pet_config=pet_config_dict,
            source_module_paths=source_pet_paths,
            source_kme_paths=source_kme_paths
        )
        admit_instance.to(device)
    except Exception as e:
        logging.error(f"Failed to instantiate AdMiTModel: {e}", exc_info=True)
        return -1.0

    # 5. Instantiate KME Feature Extractor
    logging.info("Instantiating KME Feature Extractor...")
    try:
        kme_extractor = feature_extractor.FeatureExtractor(
            model_name=config.KME_FEATURE_EXTRACTOR,
            device=device
        )
    except Exception as e:
        logging.error(f"Failed to instantiate KME Feature Extractor: {e}")
        return -1.0

    # 6. Adaptation Loop & Evaluation
    logging.info(f"Starting adaptation loop for target {target_domain_id}...")
    total_samples = 0
    correct_predictions = 0
    start_time = time.time()

    # Create iterators for both loaders
    target_iter = iter(target_loader)
    kme_feature_iter = iter(kme_feature_loader)

    progress_bar = tqdm(range(len(target_loader)), desc=f"Adapting {target_domain_id}", leave=False)

    for i in progress_bar:
        try:
            # Get batch for adaptation/inference
            x_target, y_target = next(target_iter)
            x_target, y_target = x_target.to(device), y_target.to(device)

            # Get corresponding batch for KME feature extraction
            # Assumes the loaders yield corresponding samples in the same order (shuffle=False)
            x_kme_features_input, _ = next(kme_feature_iter) # Ignore labels from this loader

            # Extract KME features
            features_target = kme_extractor.extract_features(x_kme_features_input.numpy()) # Extractor takes numpy

            if features_target is None or features_target.shape[0] != x_target.shape[0]:
                 logging.error(f"Feature extraction failed or produced wrong number of features for batch {i}. Skipping batch.")
                 continue

            # Perform AdMiT adaptation step (match, integrate, optionally tune) and get predictions
            predictions = admit_instance.adapt_batch(x_target, features_target)

            # Evaluate predictions for this batch
            _, predicted_labels = torch.max(predictions.data, 1)
            total_samples += y_target.size(0)
            correct_predictions += (predicted_labels == y_target).sum().item()

            # Optional: Log batch accuracy or other metrics
            batch_acc = (predicted_labels == y_target).sum().item() / y_target.size(0)
            progress_bar.set_postfix(batch_acc=f"{batch_acc:.3f}")

        except StopIteration:
            logging.warning(f"DataLoader ended prematurely at step {i}/{len(target_loader)}. Evaluating based on processed batches.")
            break
        except Exception as e:
            logging.error(f"Error during adaptation batch {i}: {e}", exc_info=True)
            # Decide whether to continue or stop

    end_time = time.time()
    total_time = end_time - start_time

    # 7. Calculate Final Performance
    if total_samples == 0:
        logging.error("No samples were processed for evaluation.")
        final_accuracy = 0.0
    else:
        final_accuracy = correct_predictions / total_samples

    logging.info(f"--- Adaptation Completed for Domain: {target_domain_id} ---")
    logging.info(f"  Total Time: {total_time:.2f}s")
    logging.info(f"  Processed Samples: {total_samples}")
    logging.info(f"  Accuracy: {final_accuracy:.4f}")

    return final_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Adapt using AdMiT on Target Domains")
    parser.add_argument('--target_dataset', type=str, required=True, help="Name of the target dataset (e.g., 'cifar100c', 'digits_five')")
    parser.add_argument('--target_domains', type=str, nargs='+', required=True, help="List of target domain IDs within the dataset (e.g., 'gaussian_noise_sev3' 'fog_sev3' or 'usps')")
    parser.add_argument('--pet_method', type=str, default=config.PET_METHOD, choices=['lora', 'adapter', 'vpt'], help="PET method used for source modules")
    parser.add_argument('--base_model', type=str, default=config.BASE_MODEL, help="Base model architecture")
    parser.add_argument('--source_dir', type=str, default=config.OUTPUT_DIR, help="Base directory containing saved source modules and KMEs")
    parser.add_argument('--output_dir', type=str, default=config.RESULTS_DIR, help="Directory for saving adaptation results")
    parser.add_argument('--device', type=str, default=config.DEVICE, help="Device ('cuda' or 'cpu')")
    parser.add_argument('--use_tuning', action='store_true', help="Enable test-time tuning in AdMiTModel")
    parser.add_argument('--no_tuning', action='store_false', dest='use_tuning', help="Disable test-time tuning (AdMiT-ZeroShot)")
    parser.set_defaults(use_tuning=config.ADMIT_USE_TUNING) # Default from config

    args = parser.parse_args()

    # --- Override config ---
    config.PET_METHOD = args.pet_method
    config.BASE_MODEL = args.base_model
    config.DEVICE = args.device
    config.ADMIT_USE_TUNING = args.use_tuning
    # Define source directories based on args.source_dir
    source_pet_dir = os.path.join(args.source_dir, "pet_modules")
    source_kme_dir = os.path.join(args.source_dir, "kmes")
    # Define results dir
    config.RESULTS_DIR = args.output_dir
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    utils.setup_logging()
    utils.set_seed(config.SEED) # Seed might be less critical for TTA eval but good practice

    logging.info(f"Starting target adaptation script with args: {args}")
    logging.info(f"Test-time tuning enabled: {config.ADMIT_USE_TUNING}")

    # Get number of classes based on dataset
    try:
        num_classes = config.NUM_CLASSES[args.target_dataset]
    except KeyError:
        logging.error(f"Number of classes not defined in config for dataset: {args.target_dataset}")
        exit(1)

    # Get PET config based on method
    try:
        pet_config_dict = config.PET_CONFIG[args.pet_method]
    except KeyError:
        logging.error(f"Configuration not defined in config for PET method: {args.pet_method}")
        exit(1)

    # Check if source directories exist
    if not os.path.isdir(source_pet_dir):
         logging.error(f"Source PET directory not found: {source_pet_dir}")
         exit(1)
    if not os.path.isdir(source_kme_dir):
         logging.error(f"Source KME directory not found: {source_kme_dir}")
         exit(1)

    # Loop through specified target domains
    results = {}
    for domain_id in args.target_domains:
        logging.info(f"Processing target domain: {domain_id}")

        accuracy = adapt_on_target(
            target_dataset_name=args.target_dataset,
            target_domain_id=domain_id,
            pet_method_name=args.pet_method,
            pet_config_dict=pet_config_dict,
            base_model_name=args.base_model,
            num_classes=num_classes,
            source_module_dir=source_pet_dir,
            source_kme_dir=source_kme_dir,
            device=args.device
        )
        results[domain_id] = accuracy

    logging.info("--- Adaptation Results Summary ---")
    total_acc = 0
    valid_domains = 0
    for domain, acc in results.items():
        logging.info(f"  Target Domain: {domain:<25} Accuracy: {acc:.4f}")
        if acc >= 0: # Count valid results
            total_acc += acc
            valid_domains += 1

    if valid_domains > 0:
         average_accuracy = total_acc / valid_domains
         logging.info(f"Average Accuracy across {valid_domains} domains: {average_accuracy:.4f}")
    else:
         logging.info("No valid results obtained.")

    # Optional: Save results to a file
    results_filename = f"admit_results_{args.target_dataset}_{args.pet_method}.txt"
    results_filepath = os.path.join(config.RESULTS_DIR, results_filename)
    try:
        with open(results_filepath, 'w') as f:
             f.write(f"AdMiT Adaptation Results\n")
             f.write(f"Target Dataset: {args.target_dataset}\n")
             f.write(f"PET Method: {args.pet_method}\n")
             f.write(f"Tuning Enabled: {config.ADMIT_USE_TUNING}\n")
             f.write("-" * 30 + "\n")
             for domain, acc in results.items():
                  f.write(f"{domain}: {acc:.4f}\n")
             if valid_domains > 0:
                  f.write("-" * 30 + "\n")
                  f.write(f"Average Accuracy: {average_accuracy:.4f}\n")
        logging.info(f"Results saved to {results_filepath}")
    except Exception as e:
         logging.error(f"Failed to save results to {results_filepath}: {e}")


    logging.info("Target adaptation script finished.")