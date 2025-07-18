"""
Depth Pruning - Module for removing entire transformer layers from models.

This module provides functionality to prune complete transformer layers,
which is more aggressive than neuron-level pruning but can lead to significant
efficiency gains with proper fine-tuning.
"""

import torch
from torch import nn
import logging
from typing import List, Optional, Union, Tuple, Dict, Any
from tqdm import tqdm
from transformers import PreTrainedModel

from .utils import get_model_layers, count_parameters

logger = logging.getLogger(__name__)


def validate_layer_removal_params(
    model: PreTrainedModel,
    num_layers_to_remove: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    depth_pruning_percentage: Optional[float] = None,
    layer_selection_method: str = "last"
) -> Dict[str, Any]:
    """
    Validate parameters for layer removal and return validated configuration.
    
    This function ensures that the layer removal parameters are valid and
    mutually exclusive where appropriate. It follows the same validation
    pattern as the existing MLP pruning functions.
    
    Args:
        model: Pre-trained model to validate
        num_layers_to_remove: Number of layers to remove
        layer_indices: Specific layer indices to remove
        depth_pruning_percentage: Percentage of layers to remove
        layer_selection_method: Method for selecting layers ("last", "custom")
        
    Returns:
        Dictionary with validated parameters and model info
        
    Raises:
        ValueError: If parameters are invalid or mutually exclusive
    """
    # Get model layers using existing utility
    layers = get_model_layers(model)
    if not layers:
        raise ValueError("Could not find transformer layers in the model.")
    
    total_layers = len(layers)
    
    # Count non-None parameters to ensure mutual exclusivity
    param_count = sum(1 for p in [num_layers_to_remove, layer_indices, depth_pruning_percentage] if p is not None)
    
    if param_count == 0:
        raise ValueError("Must specify one of: num_layers_to_remove, layer_indices, or depth_pruning_percentage")
    
    if param_count > 1:
        raise ValueError("Parameters num_layers_to_remove, layer_indices, and depth_pruning_percentage are mutually exclusive")
    
    # Validate layer_selection_method
    valid_methods = ["last", "custom"]
    if layer_selection_method not in valid_methods:
        raise ValueError(f"layer_selection_method must be one of {valid_methods}, got {layer_selection_method}")
    
    # Validate specific parameters
    if num_layers_to_remove is not None:
        if not isinstance(num_layers_to_remove, int) or num_layers_to_remove <= 0:
            raise ValueError("num_layers_to_remove must be a positive integer")
        if num_layers_to_remove >= total_layers:
            raise ValueError(f"Cannot remove {num_layers_to_remove} layers from model with {total_layers} layers")
    
    if depth_pruning_percentage is not None:
        if not 0 < depth_pruning_percentage < 100:
            raise ValueError("depth_pruning_percentage must be between 0 and 100")
        num_layers_to_remove = int(total_layers * depth_pruning_percentage / 100)
        if num_layers_to_remove >= total_layers:
            raise ValueError(f"depth_pruning_percentage {depth_pruning_percentage}% would remove all layers")
        if num_layers_to_remove == 0:
            raise ValueError(f"depth_pruning_percentage {depth_pruning_percentage}% would remove 0 layers")
    
    if layer_indices is not None:
        if not isinstance(layer_indices, list) or not layer_indices:
            raise ValueError("layer_indices must be a non-empty list")
        if not all(isinstance(idx, int) for idx in layer_indices):
            raise ValueError("All layer_indices must be integers")
        if not all(0 <= idx < total_layers for idx in layer_indices):
            raise ValueError(f"All layer_indices must be between 0 and {total_layers-1}")
        if len(set(layer_indices)) != len(layer_indices):
            raise ValueError("layer_indices must not contain duplicates")
        if len(layer_indices) >= total_layers:
            raise ValueError(f"Cannot remove {len(layer_indices)} layers from model with {total_layers} layers")
        
        # For custom indices, override selection method
        layer_selection_method = "custom"
        num_layers_to_remove = len(layer_indices)
    
    return {
        "total_layers": total_layers,
        "num_layers_to_remove": num_layers_to_remove,
        "layer_indices": layer_indices,
        "layer_selection_method": layer_selection_method,
        "layers": layers
    }


def select_layers_to_remove(
    total_layers: int,
    num_layers_to_remove: int,
    layer_selection_method: str,
    custom_indices: Optional[List[int]] = None
) -> List[int]:
    """
    Select which layer indices to remove based on the specified method.
    
    This function implements different strategies for selecting layers,
    similar to how neuron selection methods work in MLP pruning.
    
    Args:
        total_layers: Total number of layers in the model
        num_layers_to_remove: Number of layers to remove
        layer_selection_method: Method for selection ("last", "custom")
        custom_indices: Specific indices when method is "custom"
        
    Returns:
        List of layer indices to remove (sorted)
        
    Raises:
        ValueError: If method is invalid or parameters don't match
    """
    if layer_selection_method == "last":
        # Remove the last N layers (typically best for maintaining model performance)
        return list(range(total_layers - num_layers_to_remove, total_layers))
    
    elif layer_selection_method == "custom":
        if custom_indices is None:
            raise ValueError("custom_indices must be provided when layer_selection_method is 'custom'")
        return sorted(custom_indices)
    
    else:
        raise ValueError(f"Unknown layer_selection_method: {layer_selection_method}")


def remove_layers_from_model(
    model: PreTrainedModel,
    layer_indices_to_remove: List[int],
    show_progress: bool = True
) -> PreTrainedModel:
    """
    Remove specified layers from the model.
    
    This function performs the actual layer removal, similar to how
    prune_neuron_pairs works for MLP pruning. It modifies the model
    in-place for memory efficiency.
    
    Args:
        model: Model to modify
        layer_indices_to_remove: Sorted list of layer indices to remove
        show_progress: Whether to show progress bar
        
    Returns:
        Modified model with layers removed
    """
    layers = get_model_layers(model)
    original_layer_count = len(layers)
    
    # Create set for O(1) lookup
    indices_to_remove_set = set(layer_indices_to_remove)
    
    # Build new layer list excluding the layers to remove
    new_layers = []
    
    layer_iterator = tqdm(enumerate(layers), total=len(layers), desc="Removing layers") if show_progress else enumerate(layers)
    
    for idx, layer in layer_iterator:
        if idx not in indices_to_remove_set:
            new_layers.append(layer)
    
    # Replace the model's layers using the same logic as get_model_layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # LLaMA-style models
        model.model.layers = nn.ModuleList(new_layers)
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-style models
        model.transformer.h = nn.ModuleList(new_layers)
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        # BERT-style models
        model.encoder.layer = nn.ModuleList(new_layers)
    elif hasattr(model, 'layers'):
        # Direct layers attribute
        model.layers = nn.ModuleList(new_layers)
    else:
        raise ValueError("Could not determine model architecture for layer replacement")
    
    # Update model configuration
    if hasattr(model, 'config') and hasattr(model.config, 'num_hidden_layers'):
        model.config.num_hidden_layers = len(new_layers)
        logger.info(f"Updated model config: num_hidden_layers = {len(new_layers)}")
    
    logger.info(f"Removed {len(layer_indices_to_remove)} layers. Model now has {len(new_layers)} layers.")
    
    return model


def prune_model_depth(
    model: PreTrainedModel,
    num_layers_to_remove: Optional[int] = None,
    layer_indices: Optional[List[int]] = None,
    depth_pruning_percentage: Optional[float] = None,
    layer_selection_method: str = "last",
    show_progress: bool = True,
) -> PreTrainedModel:
    """
    Prune complete transformer layers from a model.
    
    This function removes entire transformer layers, which is more aggressive
    than neuron-level pruning but can lead to significant efficiency gains.
    The function follows the same patterns as prune_model_mlp_glu.
    
    Args:
        model: Pre-trained model to prune
        num_layers_to_remove: Number of layers to remove
        layer_indices: Specific layer indices to remove (mutually exclusive with other options)
        depth_pruning_percentage: Percentage of layers to remove (mutually exclusive with other options)
        layer_selection_method: Method for selecting layers ("last", "custom")
        show_progress: Whether to show progress during pruning
        
    Returns:
        Model with layers removed
        
    Raises:
        ValueError: If parameters are invalid or model is incompatible
    """
    # Validate all parameters
    config = validate_layer_removal_params(
        model=model,
        num_layers_to_remove=num_layers_to_remove,
        layer_indices=layer_indices,
        depth_pruning_percentage=depth_pruning_percentage,
        layer_selection_method=layer_selection_method
    )
    
    # Extract validated parameters
    total_layers = config["total_layers"]
    num_layers_to_remove = config["num_layers_to_remove"]
    layer_indices = config["layer_indices"]
    layer_selection_method = config["layer_selection_method"]
    
    logger.info(f"Starting depth pruning: removing {num_layers_to_remove} layers from {total_layers} total layers")
    
    # Select which layers to remove
    if layer_selection_method == "custom":
        layers_to_remove = layer_indices
    else:
        layers_to_remove = select_layers_to_remove(
            total_layers=total_layers,
            num_layers_to_remove=num_layers_to_remove,
            layer_selection_method=layer_selection_method
        )
    
    logger.info(f"Removing layers: {layers_to_remove} using method '{layer_selection_method}'")
    
    # Perform the actual layer removal
    model = remove_layers_from_model(
        model=model,
        layer_indices_to_remove=layers_to_remove,
        show_progress=show_progress
    )
    
    return model