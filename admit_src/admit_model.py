"""
Core AdMiT model implementation, handling module loading, matching,
integration, and optional test-time adaptation/tuning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import os
import copy
from collections import OrderedDict
from operator import itemgetter

from . import config
from . import utils
from . import kme
from . import pet_modules # To access BasePETModule and integrate_modules
# Import specific PET modules when needed for instantiation
from .pet_modules.lora import LoRA
from .pet_modules.adapter import Adapter
from .pet_modules.vpt import VPT

# --- Sharpness-Aware Minimization (SAM) Optimizer ---

class SAM(optim.Optimizer):
    """
    Wraps another optimizer (e.g., SGD, AdamW) to perform SAM updates.
    Args:
        params: Parameters to optimize.
        base_optimizer: The base optimizer class (e.g., optim.SGD).
        rho (float): Neighborhood size for SAM.
        adaptive (bool): Whether to use adaptive rho (ASAM) - not implemented here.
        **kwargs: Arguments for the base optimizer.
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")
        if not issubclass(base_optimizer, optim.Optimizer):
             raise TypeError(f"base_optimizer must be a subclass of torch.optim.Optimizer, not {type(base_optimizer)}")

        self.rho = rho
        self.adaptive = adaptive # Not used in this basic version

        # Initialize base optimizer
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults = self.base_optimizer.defaults # Store defaults for state dict

        # State for storing gradients
        self.state = {} # Using default state handling of base_optimizer

        logging.info(f"SAM Initialized with rho={rho}, Base Optimizer: {base_optimizer.__name__}")

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        """
        Performs the first SAM step: calculate and ascend towards gradient norm.
        Args:
            zero_grad (bool): Whether to zero gradients before this step.
        """
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12) # Add epsilon for stability

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale.to(p) # Compute ascent direction
                p.add_(e_w)  # Climb to the worst point e_w = rho * grad / ||grad||
                self.state[p]["e_w"] = e_w # Store e_w for the second step

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        """
        Performs the second SAM step: calculate gradient at the perturbed point
        and step using the base optimizer.
        Args:
            zero_grad (bool): Whether to zero gradients before this step.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or p not in self.state: continue
                # Ensure e_w exists and is on the correct device
                if "e_w" not in self.state[p]:
                     logging.warning(f"Parameter {p.shape} skipped in SAM second step: 'e_w' not found in state.")
                     continue
                e_w = self.state[p]["e_w"].to(p)

                p.sub_(e_w) # Move back to the original point p
                # The gradient p.grad should now be the gradient at p + e_w
                # We perform the base optimizer step using this gradient
                # self.base_optimizer.step() will use the current p.grad

        self.base_optimizer.step() # Base optimizer updates parameters using grad at p + e_w

        # Clear the stored e_w after the step? Maybe not necessary if overwritten next time.
        # for group in self.param_groups:
        #     for p in group["params"]:
        #          if p in self.state and "e_w" in self.state[p]:
        #               del self.state[p]['e_w']


        if zero_grad: self.zero_grad()

    def step(self, closure=None):
        """
        Performs a SAM optimization step.
        Args:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss. Required for LBFGS etc.
                                          but typically used for SAM's two steps.
        """
        raise NotImplementedError("SAM requires calling first_step and second_step explicitly.")

    def _grad_norm(self):
        """Computes the norm of the gradients."""
        # Use shared_device to handle gradients potentially being on multiple devices? Assume single device for now.
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        """Loads the optimizer state (for base optimizer)."""
        super().load_state_dict(state_dict)
        self.base_optimizer.load_state_dict(state_dict["base_optimizer_state"])

    def state_dict(self):
        """Returns the optimizer state (includes base optimizer state)."""
        state = super().state_dict()
        state["base_optimizer_state"] = self.base_optimizer.state_dict()
        return state

# --- AdMiT Model ---

class AdMiTModel(nn.Module):
    """
    AdMiT model orchestrator. Loads a base model, manages source PET modules
    and KMEs, performs KME matching, integrates modules, and handles adaptation.
    """
    def __init__(self, base_model, num_classes, pet_method_name, pet_config, source_module_paths, source_kme_paths):
        """
        Initializes the AdMiT model.
        Args:
            base_model (nn.Module): The pre-trained base model (e.g., ViT instance).
            num_classes (int): Number of output classes for the task.
            pet_method_name (str): Name of the PET method used ('lora', 'adapter', 'vpt').
            pet_config (dict): Configuration for the PET method.
            source_module_paths (list[str]): List of file paths to saved source PET module state dicts.
            source_kme_paths (list[str]): List of file paths to saved source KME representations.
        """
        super().__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.pet_method_name = pet_method_name
        self.pet_config = pet_config
        self.source_module_paths = source_module_paths
        self.source_kme_paths = source_kme_paths
        self.device = next(self.base_model.parameters()).device # Get device from model

        self.source_pet_states = []
        self.source_kme_reprs = []
        self.current_integrated_pet_state = None # State dict of the currently active module
        self.pet_module_class = self._get_pet_class()

        # Load source info immediately
        self._load_source_info()

        # Instantiate a temporary PET module to manage parameters during adaptation
        # We load integrated states into this temporary module
        self.temp_pet_instance = self._create_pet_instance()

        logging.info(f"AdMiTModel initialized for PET method '{pet_method_name}' with {len(self.source_pet_states)} source modules.")

    def _get_pet_class(self):
        """Gets the class corresponding to the PET method name."""
        if self.pet_method_name == 'lora':
            from .pet_modules.lora import LoRA; return LoRA
        elif self.pet_method_name == 'adapter':
            from .pet_modules.adapter import Adapter; return Adapter
        elif self.pet_method_name == 'vpt':
            from .pet_modules.vpt import VPT; return VPT
        else:
            raise ValueError(f"Unsupported PET method: {self.pet_method_name}")

    def _create_pet_instance(self):
         """Creates an instance of the appropriate PET module class."""
         # Create a deep copy of the base model config if needed by PET class
         # Note: Base model is potentially modified by PET instantiation, handle carefully
         # Maybe pass only config, not the model itself if PET class can derive structure?
         # For now, pass the base model, assuming PET class handles it appropriately.
         try:
            instance = self.pet_module_class(copy.deepcopy(self.base_model), self.pet_config)
            instance = instance.to(self.device)
            return instance
         except NotImplementedError:
             logging.error(f"Cannot create PET instance: {self.pet_method_name} is not fully implemented.")
             raise
         except Exception as e:
             logging.error(f"Error creating PET instance for {self.pet_method_name}: {e}")
             raise


    def _load_source_info(self):
        """Loads source PET module state dicts and KME representations."""
        logging.info("Loading source PET module states and KME representations...")
        if len(self.source_module_paths) != len(self.source_kme_paths):
            raise ValueError("Number of source module paths and KME paths must match.")

        for pet_path, kme_path in zip(self.source_module_paths, self.source_kme_paths):
            # Load PET state dict
            if os.path.exists(pet_path):
                try:
                    # Ensure state dict is loaded to CPU first, then moved if needed
                    pet_state = torch.load(pet_path, map_location='cpu')
                    # Basic check if it looks like a state dict
                    if isinstance(pet_state, OrderedDict):
                        self.source_pet_states.append(pet_state)
                        logging.debug(f"Loaded PET state from {pet_path}")
                    else:
                         logging.warning(f"Expected state_dict (OrderedDict) but got {type(pet_state)} from {pet_path}. Skipping.")
                except Exception as e:
                    logging.error(f"Failed to load PET state from {pet_path}: {e}")
            else:
                logging.warning(f"Source PET state file not found: {pet_path}. Skipping.")

            # Load KME representation
            kme_repr = utils.load_kme(kme_path) # utils.load_kme handles checks and logging
            if kme_repr is not None:
                 # Basic check if it has expected attributes
                 if hasattr(kme_repr, 'z') and hasattr(kme_repr, 'beta'):
                      self.source_kme_reprs.append(kme_repr)
                      logging.debug(f"Loaded KME representation from {kme_path}")
                 else:
                      logging.warning(f"Loaded object from {kme_path} does not look like a valid KME representation. Skipping.")
            else:
                 logging.warning(f"Failed to load KME from {kme_path}. Skipping corresponding source.")
                 # Remove the potentially loaded PET state if KME loading failed?
                 if len(self.source_pet_states) > len(self.source_kme_reprs):
                     self.source_pet_states.pop() # Remove the last added PET state
                     logging.warning(f"Removed PET state associated with missing KME {kme_path}")


        logging.info(f"Successfully loaded {len(self.source_pet_states)} PET states and {len(self.source_kme_reprs)} KME representations.")
        # Ensure consistency after loading
        if len(self.source_pet_states) != len(self.source_kme_reprs):
             logging.error("Inconsistency after loading: Number of PET states and KMEs do not match. Cannot proceed.")
             # Decide on error handling: raise error or try to reconcile?
             # For now, raise an error.
             raise RuntimeError("Mismatch between loaded PET states and KME representations.")


    def _integrate_pet_module(self, selected_indices, weights):
        """Integrates selected source PET modules using provided weights."""
        if not selected_indices:
            logging.warning("No modules selected for integration.")
            return None
        if len(selected_indices) != len(weights):
            logging.error(f"Number of selected indices ({len(selected_indices)}) and weights ({len(weights)}) mismatch.")
            return None

        selected_pet_states = [self.source_pet_states[i] for i in selected_indices]

        try:
            integrated_state = self.pet_module_class.integrate_modules(selected_pet_states, weights)
            logging.debug(f"Integrated state dict created from {len(selected_indices)} modules.")
            return integrated_state
        except NotImplementedError:
             logging.error(f"Integration logic for {self.pet_method_name} not implemented in {self.pet_module_class.__name__}.")
             return None
        except Exception as e:
             logging.error(f"Error during PET module integration: {e}")
             return None

    def adapt_batch(self, x_target_batch, features_target_batch):
        """
        Performs AdMiT adaptation for a single batch of target data.
        Includes KME matching, module integration, and optional tuning.

        Args:
            x_target_batch (torch.Tensor): Batch of target images (for inference/tuning).
            features_target_batch (np.ndarray): Pre-computed features for KME matching (N, feature_dim).

        Returns:
            torch.Tensor: Predictions from the adapted model for the batch.
        """
        if not self.source_kme_reprs or not self.source_pet_states:
            logging.error("Cannot adapt: No valid source KMEs or PET states loaded.")
            # Return predictions from the base model? Or raise error?
            with torch.no_grad():
                 return self.base_model(x_target_batch)

        # 1. Compute Target KME Representation
        target_kme_repr = kme.compute_kme_representation(features_target_batch)
        if target_kme_repr is None:
            logging.error("Failed to compute target KME representation for the batch. Using previous integrated module if available.")
            # Use existing self.current_integrated_pet_state if available, otherwise fallback
            if self.current_integrated_pet_state is None:
                 with torch.no_grad(): return self.base_model(x_target_batch)
            else:
                 # Load the previous state and proceed to inference/tuning
                 self.temp_pet_instance.load_pet_state_dict(self.current_integrated_pet_state)
                 integrated_state_for_tuning = self.current_integrated_pet_state # Use previous state for tuning
        else:
            # 2. Estimate Coefficients
            coeffs = kme.coefficient_estimation(target_kme_repr, self.source_kme_reprs)
            if coeffs is None:
                logging.error("Failed to estimate coefficients. Using previous integrated module if available.")
                if self.current_integrated_pet_state is None:
                     with torch.no_grad(): return self.base_model(x_target_batch)
                else:
                     self.temp_pet_instance.load_pet_state_dict(self.current_integrated_pet_state)
                     integrated_state_for_tuning = self.current_integrated_pet_state
            else:
                # 3. Select Top-M Modules
                num_to_select = min(config.ADMIT_NUM_MODULES_SELECT, len(coeffs))
                # Ensure coeffs is numpy array for argsort
                coeffs_np = np.asarray(coeffs)
                # Get indices sorted by coefficient value (descending)
                sorted_indices = np.argsort(-coeffs_np)
                selected_indices = sorted_indices[:num_to_select]
                selected_coeffs = coeffs_np[selected_indices]

                # Renormalize selected coefficients to sum to 1
                if np.sum(selected_coeffs) > 1e-6:
                    selected_coeffs_normalized = selected_coeffs / np.sum(selected_coeffs)
                else:
                    logging.warning("Selected coefficients sum to near zero. Using uniform weights.")
                    selected_coeffs_normalized = np.ones_like(selected_coeffs) / len(selected_coeffs)

                logging.info(f"Selected top {num_to_select} modules with indices: {selected_indices} and normalized weights: {selected_coeffs_normalized.round(4)}")

                # 4. Integrate Modules
                integrated_state = self._integrate_pet_module(selected_indices, selected_coeffs_normalized)
                if integrated_state is None:
                    logging.error("Failed to integrate PET modules. Using previous integrated module if available.")
                    if self.current_integrated_pet_state is None:
                         with torch.no_grad(): return self.base_model(x_target_batch)
                    else:
                         self.temp_pet_instance.load_pet_state_dict(self.current_integrated_pet_state)
                         integrated_state_for_tuning = self.current_integrated_pet_state
                else:
                    # Load the newly integrated state into the temporary PET instance
                    self.temp_pet_instance.load_pet_state_dict(integrated_state)
                    self.current_integrated_pet_state = integrated_state # Update current state
                    integrated_state_for_tuning = integrated_state # Use the new state for tuning


        # --- Inference ---
        # The self.temp_pet_instance now holds the integrated (or previous) state.
        # Perform inference using this instance (which internally calls base_model + PET modifications)
        self.temp_pet_instance.eval() # Ensure eval mode for inference
        with torch.no_grad():
            predictions_zero_shot = self.temp_pet_instance(x_target_batch.to(self.device))

        # --- Optional Tuning ---
        if config.ADMIT_USE_TUNING and integrated_state_for_tuning is not None:
            logging.debug("Performing test-time tuning on integrated module...")
            self.temp_pet_instance.train() # Set to train mode for tuning

            # Define optimizer - re-initialize for each batch? Or maintain state?
            # Paper implies adapting the current integrated module theta(t).
            # Let's optimize the parameters *within* self.temp_pet_instance.
            trainable_params = list(self.temp_pet_instance.get_pet_parameters())

            if not trainable_params:
                 logging.warning("No trainable PET parameters found in temp instance for tuning.")
                 self.temp_pet_instance.eval() # Set back to eval mode
                 return predictions_zero_shot # Return zero-shot predictions

            # Choose optimizer: SAM or standard
            if config.ADMIT_SHARPNESS_RHO > 0:
                # Use SAM optimizer
                # TODO: Determine appropriate base optimizer (SGD or AdamW) based on config?
                base_opt_class = optim.SGD if config.ADAPTATION_OPTIMIZER.lower() == 'sgd' else optim.AdamW
                optimizer = SAM(trainable_params, base_opt_class, rho=config.ADMIT_SHARPNESS_RHO, lr=config.ADAPTATION_LEARNING_RATE)
            else:
                # Use standard optimizer
                if config.ADAPTATION_OPTIMIZER.lower() == 'sgd':
                     optimizer = optim.SGD(trainable_params, lr=config.ADAPTATION_LEARNING_RATE, momentum=0.9) # Example momentum
                else:
                     optimizer = optim.AdamW(trainable_params, lr=config.ADAPTATION_LEARNING_RATE, weight_decay=0.0) # No weight decay typically for TTA

            # --- Tuning Loop (Simplified: 1 step) ---
            # Based on paper: Minimize entropy of pseudo-labels 
            # Pseudo-labels can be derived from predictions_zero_shot
            probs = torch.softmax(predictions_zero_shot.detach(), dim=1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1) # Batch entropy

            # Optional: Filter based on entropy/confidence 
            mask = torch.ones_like(entropy, dtype=torch.bool)
            if config.ADMIT_ENTROPY_THRESHOLD is not None:
                mask = entropy < config.ADMIT_ENTROPY_THRESHOLD
                if torch.sum(mask) == 0:
                    logging.warning("No samples below entropy threshold, skipping tuning for this batch.")
                    self.temp_pet_instance.eval()
                    return predictions_zero_shot

            # Calculate loss only on selected samples (e.g., mean entropy)
            loss = torch.mean(entropy[mask])

            if isinstance(optimizer, SAM):
                # SAM requires two steps
                loss.backward() # Calculate gradients at original point p
                optimizer.first_step(zero_grad=True) # Ascend to p + e_w

                # Calculate loss at the perturbed point
                predictions_perturbed = self.temp_pet_instance(x_target_batch.to(self.device)[mask]) # Re-evaluate model at p+e_w
                probs_perturbed = torch.softmax(predictions_perturbed, dim=1)
                entropy_perturbed = -torch.sum(probs_perturbed * torch.log(probs_perturbed + 1e-8), dim=1)
                loss_perturbed = torch.mean(entropy_perturbed)

                loss_perturbed.backward() # Calculate gradients at p + e_w
                optimizer.second_step(zero_grad=True) # Step using base optimizer

            else:
                # Standard optimizer step
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logging.debug(f"Tuning step completed. Loss: {loss.item():.4f}")

            # Update the current_integrated_pet_state with the tuned parameters
            self.current_integrated_pet_state = self.temp_pet_instance.pet_state_dict()

            # Get final predictions after tuning
            self.temp_pet_instance.eval()
            with torch.no_grad():
                predictions_final = self.temp_pet_instance(x_target_batch.to(self.device))
            return predictions_final

        else:
            # No tuning performed, return zero-shot predictions
            return predictions_zero_shot


    def forward(self, x):
        """
        Performs forward pass using the current integrated PET module.
        Assumes adapt_batch has been called previously to set the integrated state.
        """
        if self.current_integrated_pet_state is None:
            logging.warning("AdMiTModel forward called before adaptation. Using base model.")
            with torch.no_grad():
                return self.base_model(x)
        else:
            # Ensure the temp PET instance has the latest integrated state loaded
            try:
                 self.temp_pet_instance.load_pet_state_dict(self.current_integrated_pet_state)
                 self.temp_pet_instance.eval() # Ensure eval mode
                 with torch.no_grad():
                      return self.temp_pet_instance(x)
            except Exception as e:
                 logging.error(f"Error loading current PET state in forward pass: {e}. Falling back to base model.")
                 with torch.no_grad():
                      return self.base_model(x)


if __name__ == '__main__':
    # Example Usage requires setting up dummy base model, PET modules, KMEs etc.
    # This is more involved and better tested within the train/adapt scripts.
    print("--- AdMiT Model Basic Structure ---")
    print("Note: Full testing requires pre-trained modules and KMEs.")

    # Placeholder test setup (won't run correctly without full implementations)
    try:
        logging.basicConfig(level=logging.INFO)
        # 1. Create dummy base model
        base_model = models.vit_b_16(weights=None, num_classes=10) # Dummy ViT-Base

        # 2. Define dummy paths (these files need to be created by train_source.py)
        num_sources = 3
        os.makedirs(config.PET_MODULE_DIR, exist_ok=True)
        os.makedirs(config.KME_DIR, exist_ok=True)
        dummy_pet_paths = [os.path.join(config.PET_MODULE_DIR, f"source_{i}_lora.pt") for i in range(num_sources)]
        dummy_kme_paths = [os.path.join(config.KME_DIR, f"source_{i}_kme.pt") for i in range(num_sources)]

        # --- Need to create dummy files for this test ---
        # Example: Create dummy LoRA state and KME for testing
        # Requires LoRA implementation to be somewhat functional
        # For now, assume these files exist and are loadable

        # 3. Instantiate AdMiTModel (will likely fail if PET classes/files aren't ready)
        # admit_model = AdMiTModel(
        #     base_model=base_model,
        #     num_classes=10,
        #     pet_method_name='lora', # Assuming LoRA for now
        #     pet_config=config.PET_CONFIG['lora'],
        #     source_module_paths=dummy_pet_paths,
        #     source_kme_paths=dummy_kme_paths
        # )
        # admit_model.to(config.DEVICE)

        # 4. Create dummy batch data
        # dummy_batch = torch.randn(4, 3, 224, 224).to(config.DEVICE)
        # dummy_features = np.random.rand(4, config.KME_FEATURE_DIM) # Requires feature extractor

        # 5. Run adaptation (will fail if files/classes missing)
        # predictions = admit_model.adapt_batch(dummy_batch, dummy_features)
        # print(f"Adaptation output shape (example): {predictions.shape}")

        print("AdMiTModel structure defined. Needs PET implementations and data files for full test.")
        utils.logging.info("AdMiTModel basic structure test completed (placeholders).")

    except Exception as e:
        utils.logging.error(f"AdMiTModel basic test failed: {e}", exc_info=True)