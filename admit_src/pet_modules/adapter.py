"""
Implementation of Adapter PET module (based on Houlsby Adapters).
"""
import torch
import torch.nn as nn
import logging
from collections import OrderedDict
import types # Needed for hooks

from .base_pet import BasePETModule

class AdapterLayer(nn.Module):
    """A simple bottleneck adapter layer with residual connection."""
    def __init__(self, input_dim, bottleneck_dim):
        super().__init__()
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        self.non_linearity = nn.GELU() # Common choice
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)
        # Initialize up_proj weights to near zero (common practice)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.non_linearity(x)
        x = self.up_proj(x)
        return x + residual # Add back residual connection

class Adapter(BasePETModule):
    def __init__(self, base_model, config):
        """
        Initializes Adapter module.
        Args:
            base_model: The base model to adapt (e.g., ViT).
            config (dict): Configuration containing Adapter parameters like:
                           'bottleneck_dim',
                           'target_locations' (list of module name patterns where adapters should be inserted).
        """
        self.bottleneck_dim = config.get('bottleneck_dim', 64)
        # Target locations based on common ViT structure (adjust if needed)
        self.target_locations = config.get('target_locations', ['attn.proj_drop', 'mlp.drop']) # Insert after Attn proj and MLP output
        self._hook_handles = [] # To store hook handles for removal

        super().__init__(base_model, config) # Calls _add_pet_parameters
        logging.info(f"Adapter Module Initialized: bottleneck={self.bottleneck_dim}")
        self.print_trainable_parameters()


    def _add_pet_parameters(self):
        """Finds target locations and inserts Adapter layers using forward hooks."""
        logging.info(f"Adding Adapter layers using hooks at target locations: {self.target_locations}")
        self._pet_parameters.clear() # Ensure clean state
        self._remove_hooks() # Remove any existing hooks first

        adapter_layers_added = 0
        for name, module in self.base_model.named_modules():
            # Check if the module name ends with one of the target patterns
            if any(name.endswith(target) for target in self.target_locations):
                 try:
                    # Determine the input dimension for the adapter
                    # This usually corresponds to the output dimension of the preceding layer
                    # For ViT 'attn.proj_drop' and 'mlp.drop', input is the hidden dimension
                    # Need a robust way to get this dimension (e.g., from module inspection or config)
                    # Assuming ViT structure where input/output dim of Dropout/Linear is available
                    input_dim = -1
                    if hasattr(module, 'in_features'): # If target is Linear layer itself
                        input_dim = module.in_features
                    else: # Try to infer from preceding Linear/LayerNorm layer
                         # This part is tricky and model-specific
                         # Let's assume ViT hidden_dim can be accessed or is known
                         if hasattr(self.base_model, 'hidden_dim'):
                              input_dim = self.base_model.hidden_dim
                         elif hasattr(self.base_model, 'embed_dim'):
                              input_dim = self.base_model.embed_dim
                         else:
                              input_dim = 768 # Fallback for ViT-Base
                              logging.warning(f"Could not reliably determine input dim for Adapter at {name}. Using fallback {input_dim}")

                    if input_dim > 0:
                        logging.debug(f"  Adding Adapter hook after: {name} (input_dim={input_dim})")
                        adapter_layer = AdapterLayer(input_dim, self.bottleneck_dim).to(next(self.base_model.parameters()).device) # Move to correct device

                        # Store adapter layer parameters in the ParameterDict
                        # Use a unique key based on the insertion point
                        adapter_key = f"adapter_{name.replace('.', '_')}" # Create a valid key
                        self._pet_parameters[adapter_key] = adapter_layer

                        # Define the hook function
                        def create_hook(adapter_instance):
                            def hook_fn(module, input, output):
                                # Apply the adapter to the output of the target module
                                return adapter_instance(output)
                            return hook_fn

                        # Register the forward hook on the identified module
                        handle = module.register_forward_hook(create_hook(adapter_layer))
                        self._hook_handles.append(handle)
                        adapter_layers_added += 1

                 except Exception as e:
                     logging.error(f"Failed to add Adapter hook at {name}: {e}", exc_info=True)


        if adapter_layers_added == 0:
             logging.warning(f"Adapter._add_pet_parameters() did not find any target locations matching {self.target_locations} or failed to add hooks.")
        else:
             logging.info(f"Added {adapter_layers_added} Adapter layers via hooks.")

        # Ensure only Adapter parameters require gradients
        for param in self.base_model.parameters():
             param.requires_grad = False
        for param in self.get_pet_parameters():
             param.requires_grad = True

    def _remove_hooks(self):
        """Removes all registered forward hooks."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles = []

    def forward(self, *args, **kwargs):
        """Forward pass uses the base model, relying on the registered hooks."""
        # Ensure hooks are in place before forwarding
        if not self._hook_handles and len(self._pet_parameters) > 0:
             logging.warning("Adapter hooks seem to be missing, attempting to re-add.")
             # This indicates state inconsistency; ideally hooks persist or are managed externally
             self._add_pet_parameters()

        return self.base_model(*args, **kwargs)

    def __del__(self):
        """Ensure hooks are removed when the object is deleted."""
        self._remove_hooks()

    @staticmethod
    def integrate_modules(pet_state_dicts, weights):
        """Integrates Adapter state dicts by weighted averaging."""
        if not pet_state_dicts:
            return OrderedDict()
        if weights is None or len(weights) != len(pet_state_dicts):
            logging.error("Invalid weights for Adapter integration.")
            return pet_state_dicts[0] if pet_state_dicts else OrderedDict()

        integrated_state = OrderedDict()
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)

        all_keys = pet_state_dicts[0].keys()
        for key in all_keys:
             # Check if the key corresponds to an adapter parameter
             if "adapter_" in key and (".down_proj." in key or ".up_proj." in key):
                 try:
                    tensors = torch.stack([state[key] for state in pet_state_dicts])
                    current_weights = weights.to(tensors.device).view(-1, *([1] * (tensors.dim() - 1)))
                    weighted_sum = torch.sum(tensors * current_weights, dim=0)
                    integrated_state[key] = weighted_sum
                 except KeyError:
                      logging.warning(f"Key '{key}' not found in all Adapter state dicts during integration. Skipping.")
                 except Exception as e:
                      logging.error(f"Error integrating key '{key}' for Adapter: {e}")
                      integrated_state[key] = pet_state_dicts[0].get(key, None) # Fallback
             else:
                  # Copy non-adapter params (if any) from first dict
                  integrated_state[key] = pet_state_dicts[0].get(key, None)

             # Clean up None entries if fallback occurred
             if key in integrated_state and integrated_state[key] is None:
                 del integrated_state[key]

        return integrated_state

    def load_pet_state_dict(self, state_dict, strict=True):
         """ Overrides base to load into _pet_parameters directly. """
         # Need to handle potential key mismatches if model structure changed slightly
         # For simplicity, use standard load_state_dict
         return self._pet_parameters.load_state_dict(state_dict, strict=strict)