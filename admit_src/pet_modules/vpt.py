"""
Implementation of Visual Prompt Tuning (VPT) PET module.
Supports Shallow and Deep VPT variants.
"""
import torch
import torch.nn as nn
import logging
from collections import OrderedDict
import types # For method overriding/hooks

from .base_pet import BasePETModule

class VPT(BasePETModule):
    def __init__(self, base_model, config):
        """
        Initializes VPT module. Assumes base_model is a ViT-like structure.
        Args:
            base_model: The base model to adapt (typically a ViT).
            config (dict): Configuration containing VPT parameters like:
                           'prompt_length', 'vpt_type' ('shallow' or 'deep'),
                           'embedding_dim' (usually matches model's hidden dim).
        """
        self.prompt_length = config.get('prompt_length', 10)
        self.vpt_type = config.get('vpt_type', 'shallow').lower()
        self._original_forward = None # To store original model forward

        # Infer embedding dim from model - ViT specific logic
        self.embedding_dim = self._get_embedding_dim(base_model, config)

        super().__init__(base_model, config) # Calls _add_pet_parameters
        self._modify_model_forward() # Modify forward AFTER parameters are added

        logging.info(f"VPT Module Initialized: type={self.vpt_type}, length={self.prompt_length}, dim={self.embedding_dim}")
        self.print_trainable_parameters()

    def _get_embedding_dim(self, base_model, config):
        """Attempts to infer embedding dimension from a ViT model."""
        try:
            # Try common ViT attribute names
            if hasattr(base_model, 'hidden_dim'): return base_model.hidden_dim
            if hasattr(base_model, 'embed_dim'): return base_model.embed_dim
            if hasattr(base_model, 'config') and hasattr(base_model.config, 'hidden_size'): return base_model.config.hidden_size
            # Look inside patch_embed or cls_token if available
            if hasattr(base_model, 'patch_embed') and hasattr(base_model.patch_embed, 'proj') and isinstance(base_model.patch_embed.proj, nn.Conv2d):
                 return base_model.patch_embed.proj.out_channels
            if hasattr(base_model, 'cls_token') and isinstance(base_model.cls_token, nn.Parameter):
                 return base_model.cls_token.shape[-1]
        except Exception as e:
            logging.warning(f"Could not infer embedding dim automatically: {e}")

        # Fallback to config or default
        fallback_dim = config.get('embedding_dim', 768) # Default for ViT-Base
        logging.warning(f"Using embedding dim from config/fallback: {fallback_dim}")
        return fallback_dim

    def _get_num_layers(self, base_model):
         """Attempts to infer number of transformer layers from a ViT model."""
         try:
             if hasattr(base_model, 'blocks') and isinstance(base_model.blocks, nn.ModuleList):
                 return len(base_model.blocks)
             if hasattr(base_model, 'encoder') and hasattr(base_model.encoder, 'layer'): # HuggingFace style
                 return len(base_model.encoder.layer)
             if hasattr(base_model, 'config') and hasattr(base_model.config, 'num_hidden_layers'):
                 return base_model.config.num_hidden_layers
         except Exception as e:
             logging.warning(f"Could not infer number of layers automatically: {e}")
         # Fallback
         fallback_layers = 12 # ViT-Base default
         logging.warning(f"Using fallback number of layers: {fallback_layers}")
         return fallback_layers

    def _add_pet_parameters(self):
        """Creates learnable prompt tokens."""
        logging.info(f"Adding VPT prompts: type={self.vpt_type}, length={self.prompt_length}")
        self._pet_parameters.clear() # Ensure clean state

        if self.vpt_type == 'shallow':
            # Shallow prompts: Added only to the input sequence
            prompts = nn.Parameter(torch.zeros(1, self.prompt_length, self.embedding_dim))
            nn.init.xavier_uniform_(prompts.data) # Example initialization
            self._pet_parameters['shallow_prompts'] = prompts
        elif self.vpt_type == 'deep':
            # Deep prompts: Added to input of each transformer layer
            num_layers = self._get_num_layers(self.base_model)
            if num_layers is None:
                raise ValueError("Could not determine number of layers for Deep VPT.")
            # Need prompts for input layer + each block
            deep_prompts = nn.Parameter(torch.zeros(num_layers + 1, self.prompt_length, self.embedding_dim))
            nn.init.xavier_uniform_(deep_prompts.data)
            self._pet_parameters['deep_prompts'] = deep_prompts
        else:
            raise ValueError(f"Unknown VPT type: {self.vpt_type}")

        if not self._pet_parameters:
             logging.warning("VPT parameters were not added.")

        # Ensure only prompt parameters require gradients
        for param in self.base_model.parameters():
             param.requires_grad = False
        for param in self.get_pet_parameters():
             param.requires_grad = True


    def _modify_model_forward(self):
        """Modifies the base model's forward pass to incorporate prompts."""
        if self._original_forward is not None: # Restore first if already modified
            self.base_model.forward = self._original_forward
            self._original_forward = None

        # Store the original forward method
        self._original_forward = self.base_model.forward

        # Create a new forward method
        def vpt_forward(self_model, x):
            # This assumes a typical ViT forward structure. May need adjustments.
            batch_size = x.shape[0]
            device = x.device

            # 1. Patch Embedding + CLS token
            # Need to access these steps from the original model
            # Example: Assuming base_model has patch_embed and cls_token
            if hasattr(self_model, '_original_patch_embed_forward'): # Use original if available
                 x = self_model._original_patch_embed_forward(x)
            elif hasattr(self_model, 'patch_embed'):
                 x = self_model.patch_embed(x)
            else: raise AttributeError("Cannot find patch_embed in base model")

            if hasattr(self_model, 'cls_token'):
                 cls_tokens = self_model.cls_token.expand(batch_size, -1, -1)
                 x = torch.cat((cls_tokens, x), dim=1)
            else: raise AttributeError("Cannot find cls_token in base model")

            # 2. Add positional embedding (if applicable)
            if hasattr(self_model, 'pos_embed'):
                 # Adjust pos_embed if needed (prompts don't have explicit pos)
                 # Simple approach: add pos_embed only to original tokens
                 if self.vpt_type == 'shallow':
                      # Add prompts here
                      prompts = self._pet_parameters['shallow_prompts'].expand(batch_size, -1, -1).to(device)
                      # Insert prompts after CLS token
                      x_cls = x[:, :1, :]
                      x_patch = x[:, 1:, :]
                      x = torch.cat((x_cls, prompts, x_patch), dim=1)

                      # Add pos_embed only to cls + patch tokens
                      pos_embed_cls = self_model.pos_embed[:, :1, :]
                      pos_embed_patch = self_model.pos_embed[:, 1:(x_patch.shape[1] + 1), :] # Select appropriate slice
                      x[:, :1, :] = x[:, :1, :] + pos_embed_cls
                      x[:, (1+self.prompt_length):, :] = x[:, (1+self.prompt_length):, :] + pos_embed_patch

                 elif self.vpt_type == 'deep':
                     # Add input layer prompts (index 0)
                      prompts_input = self._pet_parameters['deep_prompts'][0:1, :, :].expand(batch_size, -1, -1).to(device)
                      x_cls = x[:, :1, :]
                      x_patch = x[:, 1:, :]
                      x = torch.cat((x_cls, prompts_input, x_patch), dim=1)

                      # Add pos_embed only to cls + patch tokens (similar to shallow)
                      pos_embed_cls = self_model.pos_embed[:, :1, :]
                      pos_embed_patch = self_model.pos_embed[:, 1:(x_patch.shape[1] + 1), :]
                      x[:, :1, :] = x[:, :1, :] + pos_embed_cls
                      x[:, (1+self.prompt_length):, :] = x[:, (1+self.prompt_length):, :] + pos_embed_patch
                 else: # No prompts, standard pos embed addition
                      x = x + self_model.pos_embed

            # Add dropout after pos embed
            if hasattr(self_model, 'pos_drop'):
                 x = self_model.pos_drop(x)

            # 3. Pass through Transformer Blocks
            if not hasattr(self_model, 'blocks'): raise AttributeError("Cannot find blocks in base model")

            for i, blk in enumerate(self_model.blocks):
                 if self.vpt_type == 'deep':
                      # Add deep prompts for this layer (index i+1)
                      prompts_deep = self._pet_parameters['deep_prompts'][i+1:i+2, :, :].expand(batch_size, -1, -1).to(device)
                      # Assume x shape is (batch, seq_len, dim)
                      # Remove previous layer's prompts, add new ones
                      x_cls = x[:, :1, :]
                      x_patch = x[:, (1+self.prompt_length):, :]
                      x = torch.cat((x_cls, prompts_deep, x_patch), dim=1)

                 x = blk(x)

            # 4. Final Layer Norm
            if hasattr(self_model, 'norm'):
                x = self_model.norm(x)

            # 5. Head (Classifier) - Use CLS token output
            # Output depends on model's head structure (e.g., take x[:, 0])
            if hasattr(self_model, 'fc_norm'): # ViT specific
                x = self_model.fc_norm(x[:, 0]) # Use CLS token
            elif hasattr(self_model, 'head'): # Common name
                 x = x[:, 0] # Assume CLS token output is needed
            else:
                 # Default: take CLS token if structure is standard ViT
                 x = x[:, 0]

            if hasattr(self_model, 'head'):
                 x = self_model.head(x)

            return x

        # Replace the model's forward method with the new one
        # Use types.MethodType to bind the new method to the instance
        self.base_model.forward = types.MethodType(vpt_forward, self)
        logging.info("Base model forward method modified for VPT.")

        # Special handling for patch_embed if deep prompts are used
        # (need original embeddings before deep prompts are added in blocks)
        if self.vpt_type == 'deep' and hasattr(self.base_model, 'patch_embed'):
             self.base_model._original_patch_embed_forward = self.base_model.patch_embed.forward


    def _restore_model_forward(self):
         """Restores the original forward method of the base model."""
         if self._original_forward is not None:
             self.base_model.forward = self._original_forward
             self._original_forward = None
             logging.info("Restored original base model forward method.")
         if hasattr(self.base_model, '_original_patch_embed_forward'):
              if hasattr(self.base_model, 'patch_embed'):
                   self.base_model.patch_embed.forward = self.base_model._original_patch_embed_forward
              delattr(self.base_model, '_original_patch_embed_forward')


    def forward(self, *args, **kwargs):
        """Forward pass uses the modified base model forward method."""
        # The logic is now inside the replaced self.base_model.forward
        return self.base_model(*args, **kwargs)

    def __del__(self):
         """Ensure original forward method is restored when object is deleted."""
         self._restore_model_forward()


    @staticmethod
    def integrate_modules(pet_state_dicts, weights):
        """Integrates VPT state dicts by weighted averaging of prompts."""
        if not pet_state_dicts:
            return OrderedDict()
        if weights is None or len(weights) != len(pet_state_dicts):
            logging.error("Invalid weights for VPT integration.")
            return pet_state_dicts[0] if pet_state_dicts else OrderedDict()

        integrated_state = OrderedDict()
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)

        all_keys = pet_state_dicts[0].keys()
        for key in all_keys:
            if "prompts" in key: # Catches 'shallow_prompts', 'deep_prompts'
                try:
                    tensors = torch.stack([state[key] for state in pet_state_dicts])
                    # Weights need broadcasting based on tensor dim
                    weight_shape = (-1,) + (1,) * (tensors.dim() - 1)
                    current_weights = weights.to(tensors.device).view(weight_shape)

                    weighted_sum = torch.sum(tensors * current_weights, dim=0)
                    integrated_state[key] = weighted_sum
                except KeyError:
                    logging.warning(f"Key '{key}' not found in all VPT state dicts during integration. Skipping.")
                except Exception as e:
                    logging.error(f"Error integrating key '{key}' for VPT: {e}")
                    integrated_state[key] = pet_state_dicts[0].get(key, None) # Fallback
            else:
                # Copy non-prompt params (if any) from first dict
                integrated_state[key] = pet_state_dicts[0].get(key, None)

            # Clean up None entries if fallback occurred
            if key in integrated_state and integrated_state[key] is None:
                del integrated_state[key]

        return integrated_state

    def load_pet_state_dict(self, state_dict, strict=True):
         """ Overrides base to load into _pet_parameters directly. """
         return self._pet_parameters.load_state_dict(state_dict, strict=strict)