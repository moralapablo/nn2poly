import numpy as np
import collections
import warnings 

from . import helpers
from . import combinatorics
from . import algorithms
# nn2poly_py.activations is used by helpers.get_taylor_series_for_layers

def nn2poly(weights_list, af_string_list, max_order=2, keep_layers=False,
            taylor_orders_param=8, precomputed_partitions=None,
            variable_names_list=None, # For named output, not in R version but good Pythonic add
            verbose=False):
    """
    Converts a neural network into its polynomial representation.

    Args:
        weights_list (list): List of NumPy arrays for weights. weights_list[l] is
                             (inputs_to_layer_l_plus_bias, outputs_of_layer_l).
        af_string_list (list): List of activation function names for each layer.
        max_order (int): Maximum desired order of the final polynomial.
        keep_layers (bool): If True, return polynomials for all intermediate layers/steps.
        taylor_orders_param (int or list/NumPy array): Taylor expansion order(s).
        precomputed_partitions (dict, optional): Output of 
                                                 helpers.obtain_partitions_with_labels.
        variable_names_list (list, optional): List of strings for variable names.
        verbose (bool): If True, print progress information.

    Returns:
        dict or list: If keep_layers is False, a dict {'labels': ..., 'values': ...}
                      for the final polynomial (values transposed to terms x neurons).
                      If keep_layers is True, a list of such dicts for all steps.
    """

    # 1. Initial Validations and Setup
    if not helpers.check_weights_dimensions(weights_list):
        raise ValueError("Weight dimensions are not compatible.")

    if len(weights_list) != len(af_string_list):
        raise ValueError("Length of weights_list must match length of af_string_list.")

    p_vars = weights_list[0].shape[0] - 1 # Number of input variables
    if p_vars <= 0:
        raise ValueError("Number of input variables (derived from first weight matrix) must be positive.")
        
    num_nn_layers = len(af_string_list) # Number of layers with activations

    processed_taylor_orders = helpers.obtain_taylor_vector(taylor_orders_param, af_string_list)
    if verbose: print(f"Processed Taylor orders for layers: {processed_taylor_orders}")

    layer_derivatives = helpers.get_taylor_series_for_layers(af_string_list, processed_taylor_orders)
    
    # q_max_final_order is the maximum order the polynomial can reach considering Taylor expansions,
    # capped by user's max_order. This is the overall q_max for partition generation.
    q_max_final_order = helpers.obtain_final_poly_order(max_order, processed_taylor_orders)
    if verbose: print(f"Effective maximum polynomial order (q_max_final_order): {q_max_final_order}")

    if precomputed_partitions is None:
        if verbose: print(f"Generating partitions for p_vars={p_vars}, q_max={q_max_final_order}...")
        all_partitions_struct = helpers.obtain_partitions_with_labels(
            p_vars, q_max_final_order, combinatorics, algorithms
        )
        if verbose: print("Partition generation complete.")
    else:
        all_partitions_struct = precomputed_partitions
    
    partition_labels_for_alg = all_partitions_struct['labels']
    partition_data_for_alg = all_partitions_struct['partitions']


    # 2. Initial Polynomial Basis (P_0: input variables and intercept)
    # Labels: intercept=(), var1=(1,), var2=(2,), ...
    current_P_labels = [tuple()] + [(i,) for i in range(1, p_vars + 1)]
    # Coefficients: Identity matrix. Each "neuron" is one input variable or the intercept.
    # Shape: (num_input_vars_plus_intercept, num_input_vars_plus_intercept)
    # current_P_values[0,:] are coeffs for intercept neuron (1 at intercept term, 0 elsewhere)
    # current_P_values[1,:] are coeffs for x1 neuron (1 at x1 term, 0 elsewhere)
    current_P_values = np.eye(p_vars + 1) 
    current_max_order = 1 # Max order of polynomials in current_P_values

    full_results_log = []
    if keep_layers:
        # Log initial state P_0 (transposed as per convention for final output)
        full_results_log.append({'description': 'P_0 (Input Basis)', 
                                 'labels': list(current_P_labels), # Ensure copy
                                 'values': current_P_values.T.copy()})

    # 3. Loop through Neural Network Layers
    # layer_idx corresponds to index in weights_list and af_string_list
    for layer_idx in range(num_nn_layers):
        if verbose: print(f"\nProcessing NN Layer {layer_idx + 1}/{num_nn_layers}...")
        
        W_this_layer = weights_list[layer_idx] # Shape: (inputs_to_W+bias, outputs_of_W)
                                               # e.g., (p_vars+1, h1) for layer 0
                                               # or (h_prev+1, h_curr) for subsequent
        af_this_layer_name = af_string_list[layer_idx]
        g_coeffs_this_layer = layer_derivatives[layer_idx]
        taylor_order_this_layer = processed_taylor_orders[layer_idx]

        # --- A. Linear Combination Step ---
        # Input: current_P_labels, current_P_values (shape: num_source_neurons x num_source_terms)
        # Output: Z_values (shape: num_dest_neurons x num_source_terms), Z_labels (same as input)
        if verbose: print(f"  Layer {layer_idx + 1}: Linear combination step.")
        
        num_dest_neurons = W_this_layer.shape[1]
        
        try:
            idx_const_in_P_labels = current_P_labels.index(tuple())
        except ValueError:
            raise ValueError("Constant term tuple() not found in current_P_labels during linear step.")

        # P_const_col are the constant terms of the polynomials for each source neuron
        P_const_col_values = current_P_values[:, idx_const_in_P_labels].reshape(-1, 1) # Shape: (num_source_neurons, 1)
        P_main_cols_values = np.delete(current_P_values, idx_const_in_P_labels, axis=1)
        # P_main_cols_labels = [l for l in current_P_labels if l != tuple()] # Not directly needed for math

        W_bias_vec = W_this_layer[0, :].reshape(1, num_dest_neurons) # Shape: (1, num_dest_neurons)
        W_main_mat = W_this_layer[1:, :] # Shape: (num_source_neurons, num_dest_neurons)

        # Calculate coefficients for main terms after linear combination
        # (W_main_mat.T @ P_main_cols_values) -> (num_dest_neurons, num_source_terms-1)
        Z_main_cols_values = W_main_mat.T @ P_main_cols_values
        
        # Calculate constant terms after linear combination
        # (W_main_mat.T @ P_const_col_values) -> (num_dest_neurons, 1)
        # W_bias_vec.T -> (num_dest_neurons, 1)
        Z_const_col_values = (W_main_mat.T @ P_const_col_values) + W_bias_vec.T
        
        # Combine constant and main terms for Z_values
        # Z_values shape: (num_dest_neurons, num_source_terms)
        Z_values = np.insert(Z_main_cols_values, idx_const_in_P_labels, Z_const_col_values.flatten(), axis=1)
        Z_labels = list(current_P_labels) # Labels remain the same through linear step

        if keep_layers:
            full_results_log.append({
                'description': f'Layer {layer_idx+1} - Output of Linear Step (Z_{layer_idx+1})',
                'labels': list(Z_labels), 
                'values': Z_values.T.copy()
            })
        
        # Update current_P_values and current_P_labels to be the output of this linear step
        current_P_values = Z_values
        current_P_labels = Z_labels
        # current_max_order does not change in linear step

        # --- B. Non-Linear Transformation Step ---
        if verbose: print(f"  Layer {layer_idx + 1}: Non-linear activation step ({af_this_layer_name}).")

        input_labels_for_nonlinear = list(current_P_labels) 
        input_values_for_nonlinear = current_P_values # Output from linear step

        # Determine max order of polynomial terms that can be generated by this non-linear step
        # current_max_order is max order of terms in input_labels_for_nonlinear
        # q_max_final_order is the global cap
        max_order_after_nonlinear = min(current_max_order * taylor_order_this_layer, q_max_final_order)
        if taylor_order_this_layer == 0 : # if activation effectively makes it constant
            max_order_after_nonlinear = 0


        # Define the set of output labels for alg_non_linear
        # These must include all monomials up to max_order_after_nonlinear
        output_labels_for_alg = [tuple()] # Start with intercept
        if max_order_after_nonlinear > 0:
            for order_val in range(1, max_order_after_nonlinear + 1):
                # Variables are 1-based: (1,), (2,), ... (p_vars)
                new_monomials = combinatorics.combinations_with_repetition(p_vars, order_val)
                output_labels_for_alg.extend(new_monomials)
        
        # Sort labels canonically: by length, then lexicographically
        output_labels_for_alg = sorted(list(set(output_labels_for_alg)), key=lambda x: (len(x), x))
        
        # Call alg_non_linear
        # alg_non_linear's current_layer is 1-based for NN layers (1=input, 2=first hidden, etc.)
        # If layer_idx = 0 (first set of W, af), this is NN layer 1.
        # alg_non_linear processes activation of layer L, so its current_layer arg is L.
        # Taylor orders in alg_non_linear are indexed by `current_layer-2`.
        # current_layer = layer_idx + 1 (if layer 0 is input, layer 1 is first hidden)
        # So if layer_idx = 0, call with current_layer = 1. (R code current_layer_R)
        # The R code calls alg_non_linear with current_layer_R + 1.
        # current_layer_R = 1..N_layers. So call is with 2..N_layers+1.
        # This means current_layer arg is "current layer number + 1" in a 1-based system.
        # Or, "which layer's activation output are we computing", starting from layer 2.
        # Let's use `layer_idx + 1` as the "NN layer number being processed".
        # If `alg_non_linear` expects `current_layer=2` for the first call after input,
        # then `layer_idx + 1` is wrong if `layer_idx` starts at 0 for the first layer.
        # The `taylor_orders` argument to `alg_non_linear` is `processed_taylor_orders`.
        # `alg_non_linear` uses `taylor_orders[current_layer-2]`.
        # If `current_layer_arg = layer_idx + 1` (e.g. 1 for first layer):
        #   `processed_taylor_orders[1-2]` is error.
        # If `current_layer_arg = layer_idx + 2` (e.g. 2 for first layer):
        #   `processed_taylor_orders[2-2 = 0]` which is `taylor_order_this_layer`. Correct.

        H_values = algorithms.alg_non_linear(
            coeffs_input=input_values_for_nonlinear,
            labels_input=input_labels_for_nonlinear,
            labels_output=output_labels_for_alg,
            taylor_orders=processed_taylor_orders, # Full list of orders for all layers
            current_layer=layer_idx + 2, # Effective NN layer number for alg_non_linear's indexing
            g=g_coeffs_this_layer,
            partitions_labels=partition_labels_for_alg,
            partitions_data=partition_data_for_alg
        )
        
        # Update current state
        current_P_labels = output_labels_for_alg
        current_P_values = H_values
        current_max_order = max_order_after_nonlinear

        if keep_layers:
            full_results_log.append({
                'description': f'Layer {layer_idx+1} - Output of Non-linear Step (H_{layer_idx+1})',
                'labels': list(current_P_labels), 
                'values': current_P_values.T.copy() # Transpose to terms x neurons
            })

        if verbose: print(f"  Layer {layer_idx + 1}: Processed. Current max order: {current_max_order}.")

    # 4. Return Value Formatting
    final_labels = current_P_labels
    final_values_transposed = current_P_values.T # Transpose to (terms x neurons)

    if variable_names_list:
        if len(variable_names_list) != p_vars:
            warnings.warn("Length of variable_names_list does not match p_vars. Ignoring names.")
        else:
            # Convert labels from integer indices to variable names
            def format_label(label_tuple):
                if not label_tuple: return "1" # Intercept
                counts = collections.Counter(label_tuple)
                terms = []
                for var_idx_1based, count in sorted(counts.items()):
                    var_name = variable_names_list[var_idx_1based - 1] # 0-indexed list
                    terms.append(f"{var_name}^{count}" if count > 1 else var_name)
                return "*".join(terms)
            
            final_labels = [format_label(l) for l in final_labels]
            if keep_layers:
                for log_entry in full_results_log:
                    log_entry['labels'] = [format_label(l) for l in log_entry['labels']]


    if keep_layers:
        return full_results_log
    else:
        return {'labels': final_labels, 'values': final_values_transposed}

```
