import numpy as np
import pandas as pd # For predict_nn2poly newdata handling
import collections
import warnings # For predict_nn2poly

def eval_poly(poly_coeffs, newdata, monomials=False):
    """
    Evaluates polynomial(s) at given data points.

    Args:
        poly_coeffs (dict): A dictionary with 'labels' and 'values'.
            - 'labels' (list): List of tuples, where each tuple represents a term's
                               variable indices (1-based). (0,) denotes intercept.
            - 'values' (np.ndarray): NumPy array of coefficients, shape
                                     (num_terms, num_polynomials).
        newdata (np.ndarray or pd.DataFrame): Data points for evaluation, shape
                                             (num_observations, num_features).
        monomials (bool): If True, returns the value of each monomial term for
                          each polynomial and observation. Otherwise, returns the
                          summed polynomial evaluations.

    Returns:
        np.ndarray: The evaluated polynomial values.
            - If monomials=False: Shape depends on squeezing (scalar, 1D, or 2D).
            - If monomials=True: Shape depends on squeezing (1D, 2D, or 3D).
    """
    # --- Input Standardization ---
    if isinstance(newdata, pd.DataFrame): # Added for predict_nn2poly convenience
        newdata = newdata.to_numpy()
    elif not isinstance(newdata, np.ndarray):
        try:
            newdata = np.array(newdata, dtype=float)
        except Exception as e:
            raise TypeError(f"newdata could not be converted to a NumPy array: {e}")

    original_newdata_ndim = newdata.ndim
    original_newdata_shape = newdata.shape

    if newdata.ndim == 0: # scalar input, assume 1 obs, 1 feature
        newdata = newdata.reshape(1, 1)
    # Handling for 1D newdata is deferred until p_features_from_labels is known.
    
    if not isinstance(poly_coeffs, dict) or 'labels' not in poly_coeffs or 'values' not in poly_coeffs:
        raise ValueError("poly_coeffs must be a dict with 'labels' and 'values'.")
    
    labels = poly_coeffs['labels']
    coefficients = poly_coeffs['values']

    if not isinstance(coefficients, np.ndarray):
        try:
            coefficients = np.array(coefficients, dtype=float)
        except Exception as e:
            raise TypeError(f"poly_coeffs['values'] could not be converted to a NumPy array: {e}")

    if coefficients.ndim == 0: 
        coefficients = coefficients.reshape(1,1)
    elif coefficients.ndim == 1: 
        coefficients = coefficients.reshape(-1, 1)
    
    num_terms_in_poly, num_polys = coefficients.shape

    if len(labels) != num_terms_in_poly:
        raise ValueError(f"Number of labels ({len(labels)}) must match number of terms "
                         f"in coefficients ({num_terms_in_poly}).")

    max_var_idx = 0
    if labels: # Only iterate if labels is not empty
        for label_tuple in labels:
            if label_tuple and label_tuple != (0,): # Not empty and not intercept
                # Ensure elements are integers before max()
                valid_indices = [l for l in label_tuple if isinstance(l, (int, np.integer)) and l != 0]
                if valid_indices:
                    current_max = max(valid_indices)
                    if current_max > max_var_idx:
                        max_var_idx = current_max
    p_features_from_labels = max_var_idx
    
    if original_newdata_ndim == 1:
        if p_features_from_labels == 1 and original_newdata_shape[0] > 1:
            newdata = newdata.reshape(-1, 1) # multiple observations of 1 feature
        else: 
            newdata = newdata.reshape(1, -1) # single observation, multiple features
    
    n_obs, p_features_from_data = newdata.shape

    if p_features_from_labels > p_features_from_data:
        raise ValueError(f"Polynomial uses variable index up to {p_features_from_labels}, "
                         f"but newdata only has {p_features_from_data} features.")

    # --- Term Evaluation (Optimized) ---
    monomial_values_all_obs_all_terms = np.ones((n_obs, num_terms_in_poly))

    for term_idx, label_tuple in enumerate(labels):
        if not label_tuple or label_tuple == (0,): # Intercept
            continue # monomial_values_all_obs_all_terms[:, term_idx] remains 1.0
        
        current_term_values_for_obs = np.ones(n_obs) # Temp for product
        for var_idx_1_based in label_tuple:
            if not isinstance(var_idx_1_based, (int, np.integer)):
                raise ValueError(f"Variable indices in labels must be integers. Found: {var_idx_1_based}")
            if not (1 <= var_idx_1_based <= p_features_from_data):
                raise ValueError(f"Variable index {var_idx_1_based} (1-based) is out of bounds "
                                 f"for newdata with {p_features_from_data} features.")
            current_term_values_for_obs *= newdata[:, var_idx_1_based - 1]
        monomial_values_all_obs_all_terms[:, term_idx] = current_term_values_for_obs
        
    # Apply coefficients via broadcasting
    # monomial_values_all_obs_all_terms is (n_obs, num_terms_in_poly)
    # coefficients is (num_terms_in_poly, num_polys)
    # term_values_with_coeffs_applied will be (n_obs, num_terms_in_poly, num_polys)
    term_values_with_coeffs_applied = (
        monomial_values_all_obs_all_terms[:, :, np.newaxis] * 
        coefficients[np.newaxis, :, :]
    )
    
    # --- Output Formatting ---
    if monomials:
        # term_values_with_coeffs_applied is (n_obs, num_terms_in_poly, num_polys)
        if num_polys == 1 and n_obs == 1:
            return term_values_with_coeffs_applied.reshape(num_terms_in_poly) 
        elif num_polys == 1 and n_obs > 1:
            return term_values_with_coeffs_applied.reshape(n_obs, num_terms_in_poly)
        elif num_polys > 1 and n_obs == 1:
            return term_values_with_coeffs_applied.reshape(num_terms_in_poly, num_polys)
        else: # num_polys > 1 and n_obs > 1
            return term_values_with_coeffs_applied
    else: # monomials = False
        summed_poly_values = np.sum(term_values_with_coeffs_applied, axis=1) # Shape: (n_obs, num_polys)
        
        if n_obs == 1 and num_polys == 1:
            return summed_poly_values[0, 0]
        elif n_obs == 1 and num_polys > 1:
            return summed_poly_values.ravel()
        elif n_obs > 1 and num_polys == 1:
            return summed_poly_values.ravel()
        else: # n_obs > 1 and num_polys > 1
            return summed_poly_values


def predict_nn2poly(nn2poly_object, newdata, monomials=False, layers=None):
    """
    Predicts values using a polynomial object generated by nn2poly.

    Args:
        nn2poly_object (dict): The polynomial object.
            - If single polynomial: {'labels': list_of_tuples, 'values': np_array_terms_x_neurons}
            - If multi-layer: {'layer_1': {'input': poly_dict, 'output': poly_dict}, ...}
        newdata (np.ndarray or pd.DataFrame): Data points for evaluation.
        monomials (bool): If True, returns monomial term values. Else, summed polynomials.
        layers (None, int, or list of int): Specifies layers for multi-layer objects.

    Returns:
        dict or np.ndarray: Predictions. Structure depends on nn2poly_object and layers.
    """
    if isinstance(newdata, pd.DataFrame): # Allow DataFrame input
        newdata = newdata.to_numpy()
    # Further newdata validation/reshaping is handled by eval_poly.

    # Case 1: nn2poly_object is a single polynomial dictionary
    if isinstance(nn2poly_object, dict) and \
       'labels' in nn2poly_object and isinstance(nn2poly_object['labels'], list) and \
       'values' in nn2poly_object and isinstance(nn2poly_object['values'], np.ndarray):
        
        if layers is not None:
            warnings.warn("'layers' argument is ignored when nn2poly_object is a single polynomial.", UserWarning)
        return eval_poly(nn2poly_object, newdata, monomials=monomials)

    # Case 2: nn2poly_object is a nested dictionary (multi-layer output)
    elif isinstance(nn2poly_object, dict) and \
         all(key.startswith('layer_') and isinstance(nn2poly_object[key], dict) for key in nn2poly_object.keys()):
        
        predictions_output = {}
        target_layer_keys = []

        if layers is None: # All layers
            sorted_layer_numbers = sorted(
                [int(k.split('_')[1]) for k in nn2poly_object.keys() if k.startswith('layer_')]
            )
            target_layer_keys = [f'layer_{num}' for num in sorted_layer_numbers]
        elif isinstance(layers, int):
            target_layer_keys = [f'layer_{layers}']
        elif isinstance(layers, list) and all(isinstance(l, int) for l in layers):
            target_layer_keys = [f'layer_{l}' for l in sorted(list(set(layers)))]
        else:
            raise ValueError("layers argument must be None, an integer, or a list of unique integers.")

        found_any_layer = False
        for layer_key in target_layer_keys:
            if layer_key not in nn2poly_object:
                warnings.warn(f"Layer {layer_key} not found in nn2poly_object. Skipping.", UserWarning)
                continue
            found_any_layer = True
            
            predictions_output[layer_key] = {}
            layer_components = nn2poly_object[layer_key] # This is {'input': poly_A, 'output': poly_B}
            
            if 'input' in layer_components and isinstance(layer_components['input'], dict):
                poly_to_eval_input = layer_components['input']
                predictions_output[layer_key]['input'] = eval_poly(
                    poly_to_eval_input, newdata, monomials=monomials
                )
            else:
                warnings.warn(f"'input' polynomial data missing or invalid for {layer_key}.", UserWarning)

            if 'output' in layer_components and isinstance(layer_components['output'], dict):
                poly_to_eval_output = layer_components['output']
                predictions_output[layer_key]['output'] = eval_poly(
                    poly_to_eval_output, newdata, monomials=monomials
                )
            else:
                warnings.warn(f"'output' polynomial data missing or invalid for {layer_key}.", UserWarning)
        
        if not found_any_layer and target_layer_keys:
            # Only raise error if specific layers were requested and none were valid.
            # If layers=None and nn2poly_object was empty (but passed initial type check), predictions_output will be {}.
            raise KeyError(f"None of the requested layers ({target_layer_keys}) found in nn2poly_object.")
        
        return predictions_output
        
    else:
        raise TypeError("nn2poly_object is not in a recognized format "
                        "(single polynomial dict or multi-layer dict with 'layer_X' keys).")

```
