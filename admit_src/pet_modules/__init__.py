# admit_src/pet_modules/__init__.py
"""
Parameter-Efficient Tuning (PET) Modules Package for AdMiT.

This package contains implementations for various PET methods like LoRA, Adapters, VPT, etc.
It also provides a base class and potentially factory functions.
"""

from .base_pet import BasePETModule
# Import concrete implementations when they are created
# from .lora import LoRA
# from .adapter import Adapter
# from .vpt import VPT

# Optional: Factory function
def get_pet_module(model, pet_method_name, pet_config, num_classes):
    """
    Factory function to create and apply a PET module to a model.
    Args:
        model (torch.nn.Module): The base model (e.g., ViT).
        pet_method_name (str): Name of the PET method ('lora', 'adapter', 'vpt').
        pet_config (dict): Configuration dictionary for the PET method.
        num_classes (int): Number of output classes for the task head.
    Returns:
        BasePETModule: The instantiated PET module integrated with the model.
    """
    if pet_method_name == 'lora':
        # from .lora import LoRA # Uncomment when implemented
        # pet_module = LoRA(model, **pet_config)
        raise NotImplementedError("LoRA module not yet implemented.")
    elif pet_method_name == 'adapter':
        # from .adapter import Adapter # Uncomment when implemented
        # pet_module = Adapter(model, **pet_config)
        raise NotImplementedError("Adapter module not yet implemented.")
    elif pet_method_name == 'vpt':
        # from .vpt import VPT # Uncomment when implemented
        # pet_module = VPT(model, **pet_config)
        raise NotImplementedError("VPT module not yet implemented.")
    else:
        raise ValueError(f"Unknown PET method: {pet_method_name}")

    # Apply the module to the model (this pattern might vary based on implementation)
    # pet_module.apply_to_model(model) # Or application happens in constructor

    # return pet_module # Return the PET module instance itself for managing parameters