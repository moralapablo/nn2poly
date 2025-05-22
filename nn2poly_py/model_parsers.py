import numpy as np
import warnings

# --- Activation Name Mapping ---
# Standardized names used by nn2poly (nn2poly_py.activations module)
# Add mappings from Keras/PyTorch names to these standard names.
# Based on R code, 'relu' is often mapped to 'softplus' for Taylor expansion.
# Other direct mappings: 'tanh', 'sigmoid', 'linear'.

_KERAS_ACTIVATION_MAP = {
    'linear': 'linear',
    'sigmoid': 'sigmoid',
    'tanh': 'tanh',
    'relu': 'softplus',  # Common approximation for Taylor-based approaches
    'softplus': 'softplus',
    # Add other Keras activation names as needed
}

_PYTORCH_ACTIVATION_MAP = {
    'Linear': 'linear', # Default if no explicit activation follows a Linear layer
    'Sigmoid': 'sigmoid',
    'Tanh': 'tanh',
    'ReLU': 'softplus',   # Common approximation
    'Softplus': 'softplus',
    # Add other PyTorch nn.Module class names for activations as needed
}


def parse_keras_model(model):
    """
    Parses a Keras-like model object to extract weights and activation functions.

    **Warning:** This is a skeletal implementation. It assumes a Keras-like API
    (e.g., model.layers, layer.get_weights(), layer.activation) but does
    NOT import tensorflow or keras. It cannot be fully tested without a
    Keras environment and may require adjustments for specific Keras versions
    or custom layers.

    Args:
        model: An object assumed to be a Keras model.

    Returns:
        dict: A dictionary with 'weights_list' (list of NumPy arrays) and
              'af_string_list' (list of activation function name strings).
              Each weight matrix in weights_list is combined (bias as first row).
    """
    weights_list = []
    af_string_list = []
    
    if not hasattr(model, 'layers'):
        warnings.warn("Model object does not have a 'layers' attribute. Cannot parse.")
        return {'weights_list': [], 'af_string_list': []}

    for layer_idx, layer in enumerate(model.layers):
        layer_weights_extracted = None
        act_func_name = 'linear' # Default activation

        # --- Try to get weights ---
        # Assuming layers with weights (like Dense) have get_weights() method
        if hasattr(layer, 'get_weights'):
            try:
                layer_weights_list = layer.get_weights()
                if layer_weights_list and len(layer_weights_list) == 2:
                    # Common case: [kernel, bias]
                    kernel = np.asarray(layer_weights_list[0]) # Ensure numpy array
                    bias = np.asarray(layer_weights_list[1])   # Ensure numpy array
                    
                    # Bias should be (1, num_units_output). Kernel is (num_units_input, num_units_output)
                    # Combined form: (1 + num_units_input, num_units_output)
                    if bias.ndim == 1:
                        bias = bias.reshape(1, -1) # Ensure bias is 2D: (1, num_units)
                    
                    if kernel.ndim != 2 or bias.ndim != 2:
                         warnings.warn(f"Layer {layer_idx}: Weight or bias is not 2D. Kernel: {kernel.ndim}D, Bias: {bias.ndim}D. Skipping layer.")
                         continue # Or handle differently

                    if kernel.shape[1] != bias.shape[1]:
                        warnings.warn(f"Layer {layer_idx}: Kernel columns ({kernel.shape[1]}) "
                                      f"don't match bias columns ({bias.shape[1]}). Skipping layer.")
                        continue

                    combined_weights = np.vstack((bias, kernel))
                    weights_list.append(combined_weights)
                    layer_weights_extracted = True # Flag that we got weights for this layer
                elif layer_weights_list: # Has weights, but not in [kernel, bias] format
                    warnings.warn(f"Layer {layer_idx} get_weights() returned "
                                  f"{len(layer_weights_list)} arrays (expected 2). Skipping weights.")
                # If no weights (e.g. pooling, dropout), layer_weights_list might be empty. These are fine to skip.
            except Exception as e: # Broad exception as we don't know the Keras API errors
                warnings.warn(f"Layer {layer_idx}: Error getting weights: {e}. Skipping weights.")
        
        # --- Try to get activation function ---
        # This activation applies to the output of the layer whose weights were just processed.
        # So, if we extracted weights, we need its corresponding activation.
        if layer_weights_extracted: # Only append activation if we appended weights
            if hasattr(layer, 'activation'):
                activation_object = layer.activation
                if hasattr(activation_object, '__name__'): # e.g. a function like tf.nn.relu
                    act_func_name_original = activation_object.__name__
                elif hasattr(activation_object, 'name'): # Keras >=2.16 might use .name for Activation instances
                    act_func_name_original = activation_object.name
                else: # Fallback for complex cases (e.g. serialized config)
                    try:
                        # For layers like Dense, activation is often in config
                        if hasattr(layer, 'get_config'):
                            layer_config = layer.get_config()
                            act_func_name_original = layer_config.get('activation', 'linear')
                        else: # Last resort
                            act_func_name_original = 'linear'
                    except Exception: # Broad exception
                        act_func_name_original = 'linear'
                        warnings.warn(f"Layer {layer_idx}: Could not determine activation name from "
                                      f"layer.activation object or get_config(). Defaulting to 'linear'.")
                
                act_func_name = _KERAS_ACTIVATION_MAP.get(act_func_name_original, 'linear')
                if act_func_name_original not in _KERAS_ACTIVATION_MAP and act_func_name_original != 'linear':
                     warnings.warn(f"Layer {layer_idx}: Keras activation '{act_func_name_original}' "
                                   f"not in map, defaulted to '{act_func_name}'.")
            
            af_string_list.append(act_func_name)
        elif not layer_weights_extracted and hasattr(layer, 'activation'):
            # This case handles layers like Activation layers themselves, which don't have weights
            # but apply an activation. nn2poly typically expects activations tied to weight layers.
            # The R code structure implies af_list corresponds to W_list.
            # If an Activation layer follows a Dense layer, the Dense layer's activation should be 'linear',
            # and then this Activation layer provides the true activation.
            # This skeletal parser is simpler: it assumes activation is part of the weighted layer.
            # If a standalone Keras Activation layer is encountered, its activation is not added
            # to af_string_list because no weights were added for it.
            pass


    if len(weights_list) != len(af_string_list):
        # This should ideally not happen if logic is correct (af added only if weights added)
        warnings.warn("Mismatch between extracted weights and activation functions. "
                      "The parser might have errors or the model structure is unexpected.")
        # Attempt to reconcile: either truncate longer list or pad shorter one with defaults.
        # Simplest: truncate to shortest, but this might lose data.
        # For now, let it return as is; user of nn2poly will get error from core checks.

    return {'weights_list': weights_list, 'af_string_list': af_string_list}


def parse_pytorch_model(model):
    """
    Parses a PyTorch-like model object to extract weights and activation functions.

    **Warning:** This is a skeletal implementation. It assumes a PyTorch-like API
    (e.g., model.children(), module.weight.data, module.bias.data) and module class names.
    It does NOT import torch. It cannot be fully tested without a PyTorch
    environment and may require adjustments for specific PyTorch versions,
    custom modules, or complex model architectures (e.g., non-sequential models,
    shared parameters). This parser is best suited for simple sequential models.

    Args:
        model: An object assumed to be a PyTorch model (e.g., an nn.Sequential).

    Returns:
        dict: A dictionary with 'weights_list' (list of NumPy arrays) and
              'af_string_list' (list of activation function name strings).
              Each weight matrix in weights_list is combined (bias as first row).
    """
    weights_list = []
    af_string_list = []

    if not hasattr(model, 'children'):
        warnings.warn("Model object does not have a 'children' attribute. Cannot parse.")
        return {'weights_list': [], 'af_string_list': []}

    try:
        modules = list(model.children())
    except Exception as e:
        warnings.warn(f"Could not retrieve model children: {e}. Cannot parse.")
        return {'weights_list': [], 'af_string_list': []}

    i = 0
    while i < len(modules):
        module = modules[i]
        module_class_name = module.__class__.__name__

        # Assuming 'Linear' layers are the ones with weights we need
        if module_class_name == 'Linear':
            # --- Try to get weights and bias from Linear layer ---
            if not (hasattr(module, 'weight') and hasattr(module.weight, 'data') and
                    hasattr(module.weight.data, 'numpy') and
                    hasattr(module, 'bias') and hasattr(module.bias, 'data') and
                    hasattr(module.bias.data, 'numpy')):
                warnings.warn(f"Module {i} ('Linear') does not have expected "
                              f"weight/bias attributes or .data.numpy() method. Skipping.")
                i += 1
                continue

            try:
                # PyTorch nn.Linear: weight is (out_features, in_features)
                # Bias is (out_features,)
                weight_matrix_np = module.weight.data.numpy().T # Transpose to (in_features, out_features)
                bias_vector_np = module.bias.data.numpy().reshape(1, -1) # (1, out_features)
                
                # Combined form: (1 + in_features, out_features)
                combined_weights = np.vstack((bias_vector_np, weight_matrix_np))
                weights_list.append(combined_weights)
            except Exception as e:
                warnings.warn(f"Module {i} ('Linear'): Error processing weights/bias: {e}. Skipping.")
                i += 1
                continue

            # --- Determine Activation Function for this Linear layer ---
            # Look at the *next* module. If it's an activation, use it. Otherwise, 'linear'.
            act_func_name = 'linear' # Default
            if i + 1 < len(modules):
                next_module = modules[i+1]
                next_module_class_name = next_module.__class__.__name__
                
                if next_module_class_name in _PYTORCH_ACTIVATION_MAP:
                    act_func_name = _PYTORCH_ACTIVATION_MAP[next_module_class_name]
                    # If we "consume" the next module as an activation, skip it in the next iteration
                    i += 1 
                # Else: No recognized activation module follows, so current Linear layer's
                # activation remains 'linear' (default).
            
            af_string_list.append(act_func_name)
        
        # For non-Linear layers that might be part of the sequence but not directly parsed for weights
        # (e.g., Dropout, BatchNorm), they are currently ignored by this logic.
        # Only Linear layers produce entries in weights_list and af_string_list.
        i += 1
        
    if len(weights_list) != len(af_string_list):
         warnings.warn("Mismatch between extracted weights and activation functions for PyTorch. "
                      "The parser might have errors or the model structure is more complex than expected.")

    return {'weights_list': weights_list, 'af_string_list': af_string_list}

```
