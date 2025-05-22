import math
import collections
import numpy as np
# from scipy.special import gamma as tgamma # Not needed if using math.factorial / math.gamma
# Assuming utils.py might have 'order' if directly translating C++, but Python's sorted is more idiomatic.
# from .utils import order 

# Helper function for canonical label generation (as per C++ logic)
def _get_canonical_label_and_comp_elements(label_original):
    """
    Computes the canonical representation of a label (an "equivalent_label_key")
    and the 'comp_elements' (unique elements of the original label, sorted by multiplicity).

    Args:
        label_original: A list or tuple of integers representing a monomial label.
                        Assumed to be 0-based indices.

    Returns:
        A tuple: (equivalent_label_key, comp_elements)
        - equivalent_label_key: A tuple of sorted 0-based ranks. This is the key.
        - comp_elements: A list of original variable indices, sorted first by
                         multiplicity in label_original (descending), then by value (ascending).
    """
    if not label_original: # Handles empty label (intercept)
        return tuple(), []

    counts = collections.Counter(label_original) # Equivalent to R's table() or C++ mset.count()
    
    # Get unique elements, sort them by multiplicity (desc) then value (asc)
    # This creates 'comp' from the C++ code.
    # R's `unique(label).sort()` gives unique elements sorted by value.
    # Then `comp = comp[order(mult, true)]` sorts these unique elements by multiplicity.
    # Python: sort items by (-multiplicity, element_value)
    sorted_unique_elements_with_counts = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    
    comp_elements = [item[0] for item in sorted_unique_elements_with_counts]
    
    if not comp_elements: # Should not happen if label_original is not empty
        return tuple(), []

    # Create a mapping from original element value in comp_elements to its 0-based rank
    element_to_rank_map = {element: rank for rank, element in enumerate(comp_elements)}
    
    # Map the original label to these 0-based ranks. This is like R's `match(label, comp)`.
    # R's match returns 1-based indices. Here, we use 0-based ranks.
    # C++: `equivalent_label = match(label, comp); equivalent_label.sort();`
    # If comp = [c1, c2, c3], then c1 is rank 0, c2 is rank 1, etc.
    # If label = [c1,c2,c1], mapped_ranks = [0,1,0]. Sorted = [0,0,1].
    mapped_ranks = [element_to_rank_map[x] for x in label_original]
    equivalent_label_key = tuple(sorted(mapped_ranks))
    
    return equivalent_label_key, comp_elements


def select_allowed_partitions(equivalent_label_key_to_find, q_previous_layer,
                              all_precomputed_labels, # List of all canonical label keys (tuples of ranks)
                              all_precomputed_partitions_data): # Parallel list to all_precomputed_labels
    """
    Selects partitions for a given canonical label, filtered by term length.

    Args:
        equivalent_label_key_to_find: Canonical form of a label (tuple of sorted ranks).
        q_previous_layer: Max length (degree) allowed for any term from the previous layer.
        all_precomputed_labels: A list of all precomputed canonical label keys.
        all_precomputed_partitions_data: A list parallel to all_precomputed_labels.
            all_precomputed_partitions_data[i] contains all partitions for all_precomputed_labels[i].
            Structure: a list of partitions, where each partition is a tuple of terms
            (terms are tuples of ranks).

    Returns:
        A list of allowed partitions. Each partition is a list of terms (tuples of ranks).
        Returns empty list if equivalent_label_key_to_find is not in all_precomputed_labels
        or if no partitions satisfy the criteria.
    """
    try:
        # Find the index of the equivalent_label_key_to_find in the global list of such keys.
        # C++ uses `match(equivalent_label, labels) - 1` for 0-based index.
        label_idx = all_precomputed_labels.index(equivalent_label_key_to_find)
    except ValueError:
        # This canonical label was not found among those for which partitions were precomputed.
        return [] 

    # partitions_for_this_specific_label is a list of partitions.
    # Each partition is a tuple of terms. Each term is a tuple of ranks.
    # Example partition: ( ((0,0),(1)), )  -- one partition, with two terms (ranks) (0,0) and (1)
    # Example list of partitions: [ ( ((0,0,1),), ),  ( ((0,0),(1)), ),  ( ((0),(0),(1)), ) ]
    partitions_for_this_specific_label = all_precomputed_partitions_data[label_idx]
    
    allowed_partitions_for_key = []
    for partition_as_tuple_of_ranked_terms in partitions_for_this_specific_label:
        is_allowed = True
        if not partition_as_tuple_of_ranked_terms: # An empty partition (no terms), e.g., ()
            # This partition is valid as all its (zero) terms satisfy len(term) <= q_previous_layer.
            pass
        else:
            for ranked_term_tuple in partition_as_tuple_of_ranked_terms:
                # len(ranked_term_tuple) is the sum of exponents of items (ranks) in that term.
                # This sum (degree of the term) must be <= q_previous_layer.
                if len(ranked_term_tuple) > q_previous_layer:
                    is_allowed = False
                    break
        
        if is_allowed:
            # Convert to list of tuples for terms, as specified by prompt ("list of terms")
            allowed_partitions_for_key.append(list(partition_as_tuple_of_ranked_terms))

    return allowed_partitions_for_key


def _calculate_multinomial_coefficient(n, m_values):
    """
    Calculates multinomial coefficient: n! / (m0! * m1! * ...).
    Args:
        n: Total number of items.
        m_values: List/tuple of counts for each distinct item type. Sum of m_values must equal n.
    """
    if n != sum(m_values):
        # This indicates a logic error upstream.
        raise ValueError(f"Sum of m_values ({sum(m_values)}) must be equal to n ({n}) for multinomial coefficient.")

    try:
        numerator = math.factorial(n)
        denominator = 1
        for m_val in m_values:
            if m_val < 0: return 0 # Factorial undefined for negative numbers.
            denominator *= math.factorial(m_val)
    except ValueError: # Should be caught by m_val < 0 for math.factorial.
        return 0 

    if denominator == 0: # Highly unlikely if m_vals are non-negative.
        return 0
    return numerator // denominator


def alg_non_linear(coeffs_input, # Numpy array: h_prev_layer x n_terms_prev_layer
                   labels_input,  # List of labels (list/tuple of original ints) for coeffs_input columns
                   labels_output, # List of labels for output polynomial columns
                   taylor_orders, # 0-indexed list/array of Taylor orders for layers L=2,3,...
                   current_layer, # Integer layer number (1=input, 2=first hidden, etc.)
                   g,             # Taylor coefficients of activation_fn for current layer (0-indexed list/array)
                                  # g[k] is (k-th derivative at 0 / k!) for Taylor expansion g(x) = sum g[k] x^k
                   partitions_labels, # List of all unique canonical_label_keys (precomputed globally)
                   partitions_data    # Parallel list to partitions_labels (precomputed globally)
                  ):
    h_prev_layer = coeffs_input.shape[0] # Number of neurons in the previous layer
    n_poly_terms_output = len(labels_output)
    
    coeffs_output = np.zeros((h_prev_layer, n_poly_terms_output))

    if current_layer < 2:
        raise ValueError("alg_non_linear is designed for current_layer >= 2 (i.e., hidden or output layers).")
    
    # taylor_orders is 0-indexed; taylor_orders[0] is for layer 2 (current_layer=2).
    q_layer = taylor_orders[current_layer - 2] # Max order for Taylor expansion at current layer

    # Determine q_previous_layer: max degree of a single term from the previous layer's output
    if not labels_input:
        q_previous_layer = 0
    elif len(labels_input) == 1 and not labels_input[0]: # Previous layer is bias-only: e.g. [[]]
        q_previous_layer = 0
    else:
        q_previous_layer = max((len(lab) for lab in labels_input if lab), default=0)

    # Pre-map input labels (as tuples) to their column indices for faster lookup
    map_label_to_idx_input = {tuple(lab): i for i, lab in enumerate(labels_input)}

    # Handle Intercept term (bias) for the current layer's output
    intercept_label_tuple = tuple() # Canonical representation of intercept label
    try:
        # Find the column index for the intercept term in the output labels.
        # Ensure that labels_output elements are converted to tuples for robust comparison if they are lists.
        idx_intercept_output = [tuple(l) for l in labels_output].index(intercept_label_tuple)
    except ValueError:
        raise ValueError("Intercept term (empty label) not found in labels_output.")

    # Get bias coefficients from the previous layer (coeffs_input[:, BIAS_COLUMN_IDX])
    bias_col_idx_input = map_label_to_idx_input.get(intercept_label_tuple)
    if bias_col_idx_input is None: # Check if previous layer had an intercept
        # If no intercept in previous layer, its contribution is as if it was zero.
        # Or, this could be an error depending on expected network structure.
        # For now, assume if not present, its coefficient is 0.
        # However, standard networks usually propagate bias.
        raise ValueError("Bias term (empty label) not found in labels_input.")
    bias_coeffs_prev_layer = coeffs_input[:, bias_col_idx_input]

    # Calculate output intercept: sum_{k=0 to q_l} g[k] * (bias_prev_layer)^k
    sum_for_intercept = np.zeros(h_prev_layer)
    for k_taylor_order in range(q_layer + 1): 
        if k_taylor_order < len(g) and g[k_taylor_order] != 0:
            term_contribution = g[k_taylor_order] * (bias_coeffs_prev_layer ** k_taylor_order)
            sum_for_intercept += term_contribution
    coeffs_output[:, idx_intercept_output] = sum_for_intercept
    
    # Loop for other polynomial terms in the output (non-intercept)
    for coeff_idx_out in range(n_poly_terms_output):
        if coeff_idx_out == idx_intercept_output:
            continue 

        current_output_label_orig = labels_output[coeff_idx_out] 
        if not current_output_label_orig: # Should be caught by above if intercept logic is robust
            continue

        # Get canonical representation and comp_elements for partition lookup
        equivalent_label_key, comp_elements_for_label = _get_canonical_label_and_comp_elements(current_output_label_orig)
        
        # select_allowed_partitions returns partitions with terms as tuples of ranks
        allowed_partitions_ranked = select_allowed_partitions(
            equivalent_label_key, q_previous_layer,
            partitions_labels, partitions_data 
        )

        if not allowed_partitions_ranked:
            continue

        # Remap ranked partitions to use original variable indices from comp_elements_for_label
        # comp_elements_for_label: [orig_var_for_rank0, orig_var_for_rank1, ...]
        allowed_partitions_original_vars = []
        for ranked_partition in allowed_partitions_ranked: # ranked_partition is a list of terms (tuples of ranks)
            original_var_list_of_terms = []
            for ranked_term_tuple in ranked_partition: # e.g. (rank0, rank0, rank1)
                # Map ranks back to original variables from current_output_label's context
                term_orig_vars = tuple(sorted([comp_elements_for_label[rank_idx] for rank_idx in ranked_term_tuple]))
                original_var_list_of_terms.append(term_orig_vars)
            allowed_partitions_original_vars.append(original_var_list_of_terms)

        # Accumulate contributions for this output coefficient
        # Loop over n_sum_order (from C++: `n` from 1 to q_l)
        # This `n_sum_order` is the number of terms (can be bias or actual P_j terms) being multiplied.
        for n_sum_order in range(1, q_layer + 1):
            if n_sum_order >= len(g) or g[n_sum_order] == 0: # g[0] is for intercept sum
                continue 

            summatory_for_g_n_term = np.zeros(h_prev_layer)
            
            for partition_of_orig_terms in allowed_partitions_original_vars:
                # partition_of_orig_terms is a list of terms: [ P1, P2, ... Pk_part ]
                # Each Pj is a tuple of original variable indices, e.g., (var1, var2)
                
                num_non_bias_terms_in_partition = len(partition_of_orig_terms)
                
                if num_non_bias_terms_in_partition > n_sum_order:
                    # This partition requires selecting more non-bias product terms than n_sum_order.
                    continue

                # Number of bias "terms" (m0) needed to make total of n_sum_order terms in product
                intercept_power_m0 = n_sum_order - num_non_bias_terms_in_partition
                
                # Counts of identical non-bias terms P_j in the partition
                term_counts_in_partition = collections.Counter(partition_of_orig_terms) 
                
                # m_values for multinomial: [m0, m1, m2, ...]
                m_values_for_multinomial = [intercept_power_m0] 
                m_values_for_multinomial.extend(term_counts_in_partition.values()) 
                
                multinomial_coeff = _calculate_multinomial_coefficient(n_sum_order, m_values_for_multinomial)
                if multinomial_coeff == 0: # Can happen if intercept_power_m0 < 0 (already checked by num_non_bias_terms > n_sum_order)
                    continue

                # Calculate product of (coeffs_input_for_term_Pj ^ count_of_Pj_in_partition)
                # This is the prod( (P_j)^m_j ) part
                prod_coeffs_powered = np.ones(h_prev_layer)
                valid_terms_in_product = True
                for term_orig_var_tuple, count_of_term_mj in term_counts_in_partition.items():
                    # term_orig_var_tuple needs to be found in labels_input (previous layer's output)
                    input_col_idx = map_label_to_idx_input.get(term_orig_var_tuple)
                    
                    if input_col_idx is None:
                        valid_terms_in_product = False
                        break 
                    
                    coeffs_for_term_Pj = coeffs_input[:, input_col_idx]
                    prod_coeffs_powered *= (coeffs_for_term_Pj ** count_of_term_mj)
                
                if not valid_terms_in_product:
                    # A term in the partition was not found in the previous layer's output labels.
                    # This might indicate an issue with partition generation or label consistency.
                    # C++ code might implicitly handle this by match returning NA -> error or 0 coeff.
                    # For safety, we skip. If this occurs, it needs investigation.
                    # print(f"Warning: Term {term_orig_var_tuple} from partition not found in labels_input. Output label: {current_output_label_orig}")
                    continue 

                bias_contribution_powered = bias_coeffs_prev_layer ** intercept_power_m0
                summatory_for_g_n_term += multinomial_coeff * prod_coeffs_powered * bias_contribution_powered
            
            coeffs_output[:, coeff_idx_out] += g[n_sum_order] * summatory_for_g_n_term
            
    return coeffs_output
