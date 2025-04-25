"""
Configuration settings for the AdMiT project.
Adapt paths and hyperparameters as needed for your setup.
"""

import os
import torch

# --- General Settings ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
SEED = 42  # Random seed for reproducibility 

# --- Paths ---
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output") # For saving models, KMEs, results
PET_MODULE_DIR = os.path.join(OUTPUT_DIR, "pet_modules")
KME_DIR = os.path.join(OUTPUT_DIR, "kmes")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")

# Ensure output directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PET_MODULE_DIR, exist_ok=True)
os.makedirs(KME_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Dataset Settings ---
# Specify names or paths relative to DATA_DIR
DATASETS = {
    "cifar100c": os.path.join(DATA_DIR, "CIFAR-100-C"),
    "digits_five": { # Example structure for Digits-Five
        "mnist": os.path.join(DATA_DIR, "digits_five", "mnist"),
        "mnistm": os.path.join(DATA_DIR, "digits_five", "mnistm"),
        "svhn": os.path.join(DATA_DIR, "digits_five", "svhn"),
        "usps": os.path.join(DATA_DIR, "digits_five", "usps"),
        "syndigits": os.path.join(DATA_DIR, "digits_five", "syndigits"),
    },
    "imagenet_c": os.path.join(DATA_DIR, "ImageNet-C"), 
    "cityscapes": os.path.join(DATA_DIR, "Cityscapes"),
    "acdc": os.path.join(DATA_DIR, "ACDC"), 
}

# --- Model Settings ---
BASE_MODEL = "vit_base_patch16_224"
# BASE_MODEL = "densenet201" 
NUM_CLASSES = { 
    "cifar100c": 100,
    "digits_five": 10,
    "imagenet_c": 1000,
    "cityscapes": 19, 
    "acdc": 19,
}
PRETRAINED_WEIGHTS = "imagenet" # Or path to custom weights

# --- PET Module Settings ---
PET_METHOD = "lora" # Default PET method ('lora', 'adapter', 'vpt') 
# Specific hyperparameters for each PET method (can be nested dicts)
PET_CONFIG = {
    "lora": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.1},
    "adapter": {"bottleneck_dim": 64}, 
    "vpt": {"prompt_length": 10, "vpt_type": "shallow"}, 
}

# --- KME Settings ---
KME_FEATURE_EXTRACTOR = "densenet201" # Model used for KME feature space 
KME_FEATURE_DIM = 1920 # Output dim of DenseNet201 features before classifier
KME_GAMMA = 0.01 # Gaussian kernel gamma 
KME_SYNTHETIC_SIZE = 10 # Size of reduced set (k in prototype) 
KME_OPTIMIZATION_STEPS = 5 # Optimization steps for synthetic data 
KME_CONSTRAINT_TYPE = 1 # 0: None, 1: Non-negative, 2: Simplex 
KME_SOLVER = 'cvxopt' # ['cvxopt', 'inv'] for coefficient estimation 
KME_EQ_CONSTRAINT = True # For coefficient estimation 
KME_NEQ_CONSTRAINT = True # For coefficient estimation 

# --- Training Settings (Source Module Pre-training) ---
SOURCE_TRAIN_EPOCHS = 10 #  Adjust as needed
SOURCE_BATCH_SIZE = 64
SOURCE_OPTIMIZER = "adamw" 
SOURCE_LEARNING_RATE = 1e-4 
SOURCE_WEIGHT_DECAY = 0.01 
SOURCE_LR_SCHEDULE = "cosine" 
SOURCE_WARMUP_EPOCHS = 5 

# --- Adaptation Settings (Target Domain) ---
TARGET_BATCH_SIZE = 128 # Default used in paper/TENT 
ADAPTATION_METHOD = "admit" # Could be 'admit', 'admit_zeroshot', 'tent', 'sar', etc.
ADAPTATION_STEPS = 1 # Number of adaptation steps per batch for methods like TENT, SAR, AdMiT-Tuning
ADAPTATION_OPTIMIZER = "sgd" # Often SGD for TTA methods
ADAPTATION_LEARNING_RATE = 1e-4 

# AdMiT Specific Adaptation Settings
ADMIT_NUM_MODULES_SELECT = 3 # M: Number of source modules to select
ADMIT_USE_TUNING = True # Whether to perform sharpness-aware tuning after integration 
ADMIT_SHARPNESS_RHO = 0.05 
ADMIT_ENTROPY_THRESHOLD = None 

# --- Device Settings ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Logging ---
LOG_LEVEL = "INFO"

print(f"Configuration loaded. Project Root: {PROJECT_ROOT}")
print(f"Using device: {DEVICE}")