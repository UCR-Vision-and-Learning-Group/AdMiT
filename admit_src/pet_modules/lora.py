"""
Implementation of LoRA (Low-Rank Adaptation) PET module.
"""
import torch
import torch.nn as nn
import logging
from collections import OrderedDict
import math
import types 

from .base_pet import BasePETModule

class LoRALayer(nn.Module):
    """Wrapper for a Linear layer to apply LoRA."""
    def __init__(self, original_layer, r, lora_alpha, lora_dropout):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = self.lora_alpha / self.r
        self.lora_dropout_p = lora_dropout

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # Create LoRA matrices as part of this layer
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5)) # Initialize A
        # B is initialized to zero

        if self.lora_dropout_p > 0.:
            self.dropout = nn.Dropout(p=self.lora_dropout_p)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        # Original layer forward pass
        original_output = self.original_layer(x)

        # LoRA path
        lora_output = self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

        return original_output + lora_output

    def get_lora_parameters(self):
         return [self.lora_A, self.lora_B]

class LoRA(BasePETModule):
    def __init__(self, base_model, config):
        """
        Initializes LoRA module.
        Args:
            base_model: The base model to adapt.
            config (dict): Configuration containing LoRA parameters like:
                           'r' (rank), 'lora_alpha', 'lora_dropout',
                           'target_modules' (list of layer types/names to apply LoRA).
        """
        self.r = config.get('r', 8)
        self.lora_alpha = config.get('lora_alpha', 16)
        self.scaling = self.lora_alpha / self.r
        self.lora_dropout = config.get('lora_dropout', 0.1)
        self.target_modules = config.get('target_modules', ['qkv', 'proj', 'fc1', 'fc2']) # ViT specific targets
        self._attached_layers = {} # Store original layers if replaced

        super().__init__(base_model, config) # This calls _add_pet_parameters

        logging.info(f"LoRA Module Initialized: r={self.r}, alpha={self.lora_alpha}, dropout={self.lora_dropout}")
        self.print_trainable_parameters()


    def _add_pet_parameters(self):
        """Finds target layers and replaces them with LoRALayer wrappers."""
        logging.info(f"Adding LoRA parameters to target modules: {self.target_modules}")
        self._pet_parameters.clear() # Ensure clean state
        self._attached_layers.clear()

        # Find candidate layers based on target_modules patterns
        modules_to_wrap = []
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear):
                if any(target_name in name for target_name in self.target_modules):
                    modules_to_wrap.append((name, module))

        if not modules_to_wrap:
            logging.warning(f"LoRA._add_pet_parameters() did not find any Linear layers matching target patterns {self.target_modules} in the base model.")
            return

        # Replace layers and store LoRA parameters
        for name, original_layer in modules_to_wrap:
            logging.debug(f"  Applying LoRA to layer: {name} (in_features={original_layer.in_features}, out_features={original_layer.out_features})")

            # Create the LoRA wrapper
            lora_wrapper = LoRALayer(original_layer, self.r, self.lora_alpha, self.lora_dropout)

            # Replace the original layer in the base model
            name_parts = name.split('.')
            parent_module = self.base_model
            for part in name_parts[:-1]:
                parent_module = getattr(parent_module, part)
            setattr(parent_module, name_parts[-1], lora_wrapper)

            # Store LoRA parameters in the ParameterDict using the wrapper's name
            self._pet_parameters[f"{name}.lora_A"] = lora_wrapper.lora_A
            self._pet_parameters[f"{name}.lora_B"] = lora_wrapper.lora_B
            self._attached_layers[name] = original_layer # Keep track of original if needed later

        # Ensure only LoRA parameters require gradients
        for param in self.base_model.parameters():
             param.requires_grad = False
        for param in self.get_pet_parameters():
             param.requires_grad = True


    def forward(self, *args, **kwargs):
        """Forward pass uses the modified base model with LoRALayers."""
        return self.base_model(*args, **kwargs)

    @staticmethod
    def integrate_modules(pet_state_dicts, weights):
        """Integrates LoRA state dicts by weighted averaging."""
        if not pet_state_dicts:
            return OrderedDict()
        if weights is None or len(weights) != len(pet_state_dicts):
            logging.error("Invalid weights for LoRA integration.")
            return pet_state_dicts[0] if pet_state_dicts else OrderedDict()

        integrated_state = OrderedDict()
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)

        # Average each parameter
        all_keys = pet_state_dicts[0].keys()
        for key in all_keys:
            if key.endswith(".lora_A") or key.endswith(".lora_B"):
                try:
                    tensors = torch.stack([state[key] for state in pet_state_dicts])
                    current_weights = weights.to(tensors.device).view(-1, *([1] * (tensors.dim() - 1)))
                    weighted_sum = torch.sum(tensors * current_weights, dim=0)
                    integrated_state[key] = weighted_sum
                except KeyError:
                    logging.warning(f"Key '{key}' not found in all LoRA state dicts during integration. Skipping.")
                except Exception as e:
                    logging.error(f"Error integrating key '{key}' for LoRA: {e}")
                    integrated_state[key] = pet_state_dicts[0].get(key, None) # Fallback
            else:
                 # Copy non-LoRA params (if any) from first dict
                 integrated_state[key] = pet_state_dicts[0].get(key, None)

            # Clean up None entries if fallback occurred
            if key in integrated_state and integrated_state[key] is None:
                del integrated_state[key]

        return integrated_state

    def load_pet_state_dict(self, state_dict, strict=True):
        """ Overrides base to load into _pet_parameters directly. """
        return self._pet_parameters.load_state_dict(state_dict, strict=strict)