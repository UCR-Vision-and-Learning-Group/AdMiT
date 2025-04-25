"""
Abstract Base Class for Parameter-Efficient Tuning (PET) modules.
"""

import abc
import torch
import torch.nn as nn
import logging
from collections import OrderedDict

class BasePETModule(nn.Module, abc.ABC):
    """
    Abstract base class for all PET modules (LoRA, Adapter, VPT, etc.).
    Ensures a common interface for training, saving, loading, and integration.
    """
    def __init__(self, base_model, config):
        super().__init__()
        # It's often better practice to modify the model passed in,
        # rather than storing a reference if the PET module itself is the main
        # nn.Module to be used forward. 
        # Storing the reference allows interaction with the base model structure.
        self.base_model = base_model
        self.config = config
        # Use ParameterDict to properly register PET parameters with PyTorch
        self._pet_parameters = nn.ParameterDict()

        logging.info(f"Initializing {self.__class__.__name__} with config: {config}")
        # Call the subclass implementation to add parameters and modify the model
        self._add_pet_parameters()

    @abc.abstractmethod
    def _add_pet_parameters(self):
        """
        Identifies layers in the base_model and adds the specific PET parameters
        (e.g., LoRA matrices, Adapter layers, prompt tokens) to self._pet_parameters
        and potentially modifies the base_model (e.g., via hooks or replacing layers).
        This method MUST be implemented by subclasses.
        It should also handle setting requires_grad flags (freeze base model, unfreeze PET params).
        """
        pass

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        """
        Defines how the PET module interacts with the base model's forward pass.
        This might involve:
        1. Directly calling the modified base_model if modifications are done in-place or via hooks.
        2. Manually applying PET parameters if they haven't modified the base_model directly.
        Implementation depends heavily on the specific PET method and how _add_pet_parameters modifies the model.
        This method MUST be implemented by subclasses.
        """
        # Example: If modifications are handled by hooks/layer replacement:
        # return self.base_model(*args, **kwargs)
        raise NotImplementedError

    def get_pet_parameters(self):
        """
        Returns an iterator over the parameters specifically added by this PET module.
        """
        # Ensure gradients are enabled only for PET parameters if needed during training
        # This should ideally be handled after module initialization or before training loop
        return self._pet_parameters.parameters()

    def pet_state_dict(self):
        """
        Returns the state dictionary containing only the PET parameters.
        """
        return self._pet_parameters.state_dict()

    def load_pet_state_dict(self, state_dict, strict=True):
        """
        Loads the PET parameters from a state dictionary into the _pet_parameters dict.
        """
        # It's crucial that the keys in the loaded state_dict match the keys
        # created in _add_pet_parameters (e.g., "adapter_module_name.down_proj.weight")
        return self._pet_parameters.load_state_dict(state_dict, strict=strict)

    @staticmethod
    @abc.abstractmethod
    def integrate_modules(pet_state_dicts, weights):
        """
        Integrates multiple PET modules of the *same type* based on weights.
        This is crucial for AdMiT's module integration step[cite: 23, 85, 110].
        Args:
            pet_state_dicts (list[OrderedDict]): A list of state dicts from PET modules of the same type.
            weights (torch.Tensor or list[float]): Weights for each module, summing to 1.
        Returns:
            OrderedDict: The integrated state dictionary for the combined PET module.
        This method MUST be implemented by subclasses.
        """
        pass

    def print_trainable_parameters(self):
        """Prints the number of trainable parameters in the PET module."""
        pet_params = sum(p.numel() for p in self.get_pet_parameters() if p.requires_grad)
        try: # Attempt to count base model params for comparison
             total_params = sum(p.numel() for p in self.base_model.parameters())
             percentage = 100 * pet_params / total_params if total_params > 0 else 0
             logging.info(f"Trainable PET parameters ({self.__class__.__name__}): {pet_params:,} ({percentage:.4f}% of base model)")
        except Exception:
             logging.info(f"Trainable PET parameters ({self.__class__.__name__}): {pet_params:,}")