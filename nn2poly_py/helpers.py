import numpy as np
import warnings # For obtain_final_poly_order warning
from . import activations # To use get_taylor_coefficients
from . import combinatorics # To use its functions for obtain_partitions_with_labels
# algorithms module will be passed as an argument for obtain_partitions_with_labels

def check_weights_dimensions(weights_list):
    """
    Checks if the dimensions of consecutive weight matrices in a list are compatible.
    For each matrix from the second one, checks if matrix.shape[0] == previous_matrix.shape[1] + 1 (for bias).

    Args:
        weights_list (list): A list of NumPy arrays, where each array is a weight matrix.
                             weights_list[0] is W_1 (input to layer 1),
                             weights_list[1] is W_2 (layer 1 to layer 2), etc.
                             Dimensions are typically (n_inputs_to_layer, n_outputs_of_layer) or
                             (n_outputs_of_layer, n_inputs_to_layer) depending on convention.
                             The R code implies W_l has dimensions (h_{l-1}+1, h_l).
                             So, W_l.shape[0] = h_{l-1}+1 and W_l.shape[1] = h_l.
                             Thus, W_{l+1}.shape[0] should be h_l + 1.
                             This means current_matrix.shape[0] == previous_matrix.shape[1] + 1.

    Returns:
        bool: True if dimensions are compatible, False otherwise.
    """
    if not weights_list or len(weights_list) < 2:
        return True # Or False, or raise error, depending on expected minimum length.
                    # R code returns TRUE for single matrix. Let's match.

    for i in range(1, len(weights_list)):
        previous_matrix = weights_list[i-1]
        current_matrix = weights_list[i]
        if not isinstance(previous_matrix, np.ndarray) or not isinstance(current_matrix, np.ndarray):
            raise TypeError("All elements in weights_list must be NumPy arrays.")
        if current_matrix.ndim != 2 or previous_matrix.ndim != 2:
            # This check is not in R code but good practice. R matrices are 2D.
            warnings.warn("Weight matrices should be 2-dimensional.") 
            # R code would proceed with matrix[[1]] vs matrix[[2]] if they are lists of vectors for example
            # For simplicity, let's assume they are proper 2D arrays.

        # R: `ncol(weights_list[[i-1]]) != nrow(weights_list[[i]]) - 1`
        # Python: previous_matrix.shape[1] (ncol) != current_matrix.shape[0] (nrow) - 1
        if previous_matrix.shape[1] != current_matrix.shape[0] - 1:
            return False
    return True

def obtain_taylor_vector(taylor_orders_param, af_string_list):
    """
    Generates a NumPy array of Taylor orders for each layer based on activation functions.

    Args:
        taylor_orders_param (int or list/NumPy array):
            - If int: The Taylor order to use for non-linear activation functions.
            - If list/NumPy array: Specific Taylor orders for each layer.
        af_string_list (list): A list of strings, where each string is the name
                               of an activation function for a layer (e.g., 'tanh', 'linear').

    Returns:
        numpy.ndarray: A 1D NumPy array of integers representing the Taylor order for each layer.
                       For 'linear' activation, the order is 1.
    
    Raises:
        ValueError: If type or length mismatches occur.
    """
    if not isinstance(af_string_list, list):
        raise TypeError("af_string_list must be a list of strings.")
    if not af_string_list:
        return np.array([], dtype=int) # No layers, no orders.

    num_layers = len(af_string_list)
    result_orders = np.zeros(num_layers, dtype=int)

    if isinstance(taylor_orders_param, int):
        if taylor_orders_param < 0:
            raise ValueError("Scalar taylor_orders must be non-negative.")
        for i in range(num_layers):
            if af_string_list[i].lower() == 'linear':
                result_orders[i] = 1
            else:
                result_orders[i] = taylor_orders_param
    elif isinstance(taylor_orders_param, (list, np.ndarray)):
        if len(taylor_orders_param) != num_layers:
            raise ValueError(f"Length of taylor_orders ({len(taylor_orders_param)}) "
                             f"must match length of af_string_list ({num_layers}).")
        for i in range(num_layers):
            order_val = taylor_orders_param[i]
            if not isinstance(order_val, int) or order_val < 0 : # Also check np.integer if numpy array
                 if isinstance(order_val, np.integer) and order_val >=0:
                     pass # it's a numpy int, like np.int64
                 else:
                    raise ValueError(f"All Taylor orders must be non-negative integers. Found: {order_val}")

            if af_string_list[i].lower() == 'linear':
                # R code behavior: If af is linear, order is 1. If user provides different for linear, it's overridden.
                result_orders[i] = 1
            else:
                result_orders[i] = int(order_val)
    else:
        raise TypeError("taylor_orders must be an int, list, or NumPy array.")

    return result_orders


def get_taylor_series_for_layers(af_string_list, taylor_orders_vector):
    """
    Obtains Taylor series coefficients for each layer.

    Args:
        af_string_list (list): List of activation function names (strings).
        taylor_orders_vector (numpy.ndarray): NumPy array of Taylor orders for each layer.

    Returns:
        list: A list of NumPy arrays, where each array contains the Taylor coefficients
              [c0, c1, ..., c_order] for the corresponding layer's activation function.
              Coefficients are f^(k)(0)/k! by default from get_taylor_coefficients.
    """
    if len(af_string_list) != len(taylor_orders_vector):
        raise ValueError("Mismatch in length between activation function list and Taylor orders vector.")

    derivatives_list = []
    for i in range(len(af_string_list)):
        func_name = af_string_list[i]
        order = taylor_orders_vector[i]
        
        # Assuming expansion around 0, as is common for Taylor series in NNs
        # unless specified otherwise by context (R code implies around 0).
        # The get_taylor_coefficients defaults to around=0.
        coeffs = activations.get_taylor_coefficients(func_name, int(order))
        derivatives_list.append(coeffs)
        
    return derivatives_list


def obtain_final_poly_order(max_order, taylor_orders_vector):
    """
    Calculates the final effective polynomial order.

    Args:
        max_order (int): A user-defined maximum order cap.
        taylor_orders_vector (numpy.ndarray): NumPy array of Taylor orders for each layer.

    Returns:
        int: The minimum of (product of Taylor orders, max_order).
    """
    if not isinstance(max_order, int) or max_order < 0:
        raise ValueError("max_order must be a non-negative integer.")
    if not isinstance(taylor_orders_vector, np.ndarray) or not np.issubdtype(taylor_orders_vector.dtype, np.integer):
        raise ValueError("taylor_orders_vector must be a NumPy array of integers.")
    if np.any(taylor_orders_vector < 0): # Orders should be non-negative, usually >= 1 for effective layers.
        raise ValueError("Taylor orders in the vector must be non-negative.")

    if taylor_orders_vector.size == 0: # No layers or empty taylor_orders_vector
        # Product of empty set is 1. If this means "input layer only", order is 1 (input itself).
        # Or 0 if no variables. Let's assume if empty, it's 1 for safety, or cap by max_order.
        # R code `prod(NULL)` is 1.
        product_of_orders = 1
    else:
        product_of_orders = np.prod(taylor_orders_vector)

    if product_of_orders < 1: # e.g. if any order was 0 and vector not empty
        product_of_orders = 1 # Default to order 1 if product is less than 1 (e.g. contains a 0)

    final_order = min(int(product_of_orders), max_order)

    if product_of_orders > max_order:
        warnings.warn(f"Product of Taylor orders ({product_of_orders}) "
                      f"exceeds max_order ({max_order}). "
                      f"Final polynomial order capped at {max_order}.")
    elif product_of_orders < max_order and product_of_orders > 0 and taylor_orders_vector.size > 0 : # check product_of_orders > 0
        # This warning is from the R code.
        # "prod_taylor < max_order" does not seem like a typical warning condition unless
        # max_order was meant to be achieved. The R code has this warning.
        warnings.warn(f"Product of Taylor orders ({product_of_orders}) "
                      f"is less than specified max_order ({max_order}). "
                      f"Effective polynomial order will be {product_of_orders}.")
    
    return final_order


def obtain_partitions_with_labels(p_variables, q_max_poly_order,
                                  combinatorics_module, algorithms_module):
    """
    Generates unique canonical labels and their corresponding partitions,
    structured for use with alg_non_linear.

    Args:
        p_variables (int): Number of input variables to the entire polynomial model.
        q_max_poly_order (int): Maximum degree of any monomial to consider for label generation.
        combinatorics_module: The nn2poly_py.combinatorics module.
        algorithms_module: The nn2poly_py.algorithms module (for _get_canonical_label_and_comp_elements).

    Returns:
        dict: A dictionary with two keys:
            'labels': A list of unique `equivalent_label_key` tuples. Each key is a
                      tuple of sorted 0-based ranks, representing a canonical monomial form.
            'partitions': A list parallel to 'labels'. Each element `partitions[i]`
                          contains all partitions for `labels[i]`. These partitions are
                          lists of tuples of terms, where terms are tuples of ranks
                          relative to the `comp_elements` of the original monomial that
                          produced `labels[i]`.
    """
    unique_equivalent_labels = []
    partitions_for_unique_labels = []
    
    # Memoize equivalent labels to avoid redundant computations if multiple original monomials
    # map to the same canonical form (though _get_canonical_label_and_comp_elements handles this).
    # The main purpose here is to store partitions only for unique canonical forms.
    seen_equivalent_labels_set = set()

    for q_degree in range(1, q_max_poly_order + 1):
        # Generate all original monomials (combinations of variables) of degree q_degree
        # These are like (var_idx1, var_idx2, ...) e.g., (0,0,1) for x0^2*x1
        original_monomials = combinatorics_module.combinations_with_repetition(p_variables, q_degree)
        
        for original_monomial_tuple in original_monomials:
            # Get the canonical form (equivalent_label_key) and the mapping (comp_elements)
            # for this specific original_monomial_tuple.
            eq_label_key, comp_elements = \
                algorithms_module._get_canonical_label_and_comp_elements(original_monomial_tuple)

            if eq_label_key not in seen_equivalent_labels_set:
                seen_equivalent_labels_set.add(eq_label_key)
                unique_equivalent_labels.append(eq_label_key)
                
                current_monomial_all_k_partitions_ranked = []
                # Now, generate all partitions for this *original_monomial_tuple*
                # into k_parts (from 1 to len(original_monomial_tuple)).
                # The terms in these partitions must be converted to ranks using comp_elements.
                
                element_to_rank_map = {element: rank for rank, element in enumerate(comp_elements)}

                for k_parts in range(1, len(original_monomial_tuple) + 1):
                    # partitions_of_orig_mono: list of [sub_part1, sub_part2, ...]
                    # where sub_part is [orig_el1, orig_el2]
                    partitions_of_orig_mono = combinatorics_module.multiset_partitions(
                        list(original_monomial_tuple), k_parts
                    )
                    
                    for single_partition_orig_elements in partitions_of_orig_mono:
                        # single_partition_orig_elements is like [[7,7], [5]]
                        # Convert this to ranks: [[0,0], [1]] if 7->0, 5->1
                        ranked_partition_terms = []
                        for sub_part_orig in single_partition_orig_elements:
                            # sub_part_orig is [7,7] or [5]
                            # Ranks must be sorted within the term for canonical term representation
                            ranked_term = tuple(sorted([element_to_rank_map[el] for el in sub_part_orig]))
                            ranked_partition_terms.append(ranked_term)
                        
                        # The overall partition (list of ranked terms) should also be sorted
                        # to ensure canonical representation of the partition itself.
                        # The output of multiset_partitions is already sorted by default.
                        # Here, ranked_partition_terms come from sorted sub-parts, and the list of these
                        # terms should also be sorted. MultisetPartitions class sorts the list of parts.
                        current_monomial_all_k_partitions_ranked.append(tuple(ranked_partition_terms))
                
                partitions_for_unique_labels.append(current_monomial_all_k_partitions_ranked)

    return {'labels': unique_equivalent_labels, 'partitions': partitions_for_unique_labels}

```
