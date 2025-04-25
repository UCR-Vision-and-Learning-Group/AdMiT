"""
Utility functions for AdMiT.
"""

import logging
import random
import numpy as np
import torch
import sys
import os
from . import config # Import config from the same package

def setup_logging():
    """Configures logging for the project."""
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(levelname)s - %(module)s - %(message)s',
                        stream=sys.stdout) # Log to stdout
    logging.info(f"Logging level set to {config.LOG_LEVEL}")

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Set random seed to {seed}")

# --- Add other utility functions as needed ---
# E.g., functions for saving/loading models, KMEs, results,
# calculating accuracy, etc.

# Example placeholder for saving KME data
def save_kme(kme_data, filepath):
    """Saves KME data (e.g., z vectors, beta weights) to a file."""
    try:
        # Assuming kme_data is a dictionary or object that can be pickled
        torch.save(kme_data, filepath)
        logging.info(f"Saved KME data to {filepath}")
    except Exception as e:
        logging.error(f"Error saving KME data to {filepath}: {e}")

# Example placeholder for loading KME data
def load_kme(filepath):
    """Loads KME data from a file."""
    try:
        if not os.path.exists(filepath):
            logging.error(f"KME file not found: {filepath}")
            return None
        kme_data = torch.load(filepath)
        logging.info(f"Loaded KME data from {filepath}")
        return kme_data
    except Exception as e:
        logging.error(f"Error loading KME data from {filepath}: {e}")
        return None


# Initialize logging and set seed when module is imported
setup_logging()
set_seed(config.SEED)