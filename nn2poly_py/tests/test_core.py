import pytest
import numpy as np
from nn2poly_py import core, helpers, combinatorics, algorithms # For precomputed_partitions

# Test data corresponding to R's testing_helper_1()
WEIGHTS_LIST_DATA = [
    np.ones((3, 2)),  # Layer 1: W_shape (inputs_inc_bias, outputs) = (2+1, 2)
    np.ones((3, 2)),  # Layer 2: W_shape (2+1, 2)
    np.ones((3, 1))   # Layer 3: W_shape (2+1, 1)
]
AF_STRING_LIST_DATA = ["softplus", "softplus", "linear"] # Activations for layer 1, 2, 3
TAYLOR_ORDERS_DATA = np.array([2, 2, 1]) # Taylor orders for af_list[0], af_list[1], af_list[2]
MAX_ORDER_DATA = 3

# Precompute partitions once for the test function if needed by core.nn2poly
# core.nn2poly calculates p_vars from weights_list[0].shape[0]-1. Here, 3-1=2.
# q_max_final_order is calculated by core.nn2poly using helpers.obtain_final_poly_order.
# For MAX_ORDER_DATA=3, TAYLOR_ORDERS_DATA=[2,2,1], product is 4. So q_max_final_order = min(4,3) = 3.
PRECOMPUTED_PARTITIONS_DATA = helpers.obtain_partitions_with_labels(
    p_variables=2, # Derived from WEIGHTS_LIST_DATA[0]
    q_max_poly_order=MAX_ORDER_DATA, # This should be q_max_final_order
    combinatorics_module=combinatorics,
    algorithms_module=algorithms
)

def test_nn2poly_algorithm_precomputed_example():
    """
    Corresponds to R's 'nn2poly_algorithm against precomputed example'.
    Uses the R test's specific weights, activations, and Taylor orders.
    """
    
    result_log = core.nn2poly(
        weights_list=WEIGHTS_LIST_DATA,
        af_string_list=AF_STRING_LIST_DATA,
        max_order=MAX_ORDER_DATA,
        keep_layers=True,
        taylor_orders_param=TAYLOR_ORDERS_DATA, # Passed as specific orders for each layer
        precomputed_partitions=PRECOMPUTED_PARTITIONS_DATA,
        variable_names_list=None, # Using default integer tuple labels for this test
        verbose=False
    )

    # --- Verification 1: Order of the last term in the final polynomial ---
    # The final polynomial is the last entry in the log.
    # Log structure: P0, Z1, H1, Z2, H2, Z3, H3 (H3 is output of non-linear for layer 3)
    # Layer 3's activation is 'linear' with Taylor order 1.
    # The non-linear step for 'linear' activation (order 1) preserves max_order.
    assert len(result_log) == 1 + 2 * len(WEIGHTS_LIST_DATA) # P0 + 2 steps per layer
    
    final_poly_step_log_entry = result_log[-1] # This is H_3
    final_labels = final_poly_step_log_entry['labels'] # Labels are sorted by length, then lexicographically

    # The highest order term will be among the last labels.
    # Example: if max order is 3, labels could be (1,1,1), (1,1,2), (1,2,2), (2,2,2)
    # len() of these tuples gives the order.
    max_len_found = 0
    if final_labels:
        max_len_found = len(final_labels[-1])
        # Verify all labels conform to this or less
        for lab in final_labels:
            if len(lab) > max_len_found: # Should not happen if sorted by length
                 max_len_found = len(lab)

    # The R test `expect_equal(length(results_r[[length(results_r)]]$labels[[n_terms]]), 3)`
    # checks the length (number of variables) of the last term's label.
    # `n_terms` in R code means `length(results_r[[length(results_r)]]$labels)`.
    # So it's the order of the very last label in the sorted list.
    # For max_order=3, p_vars=2, this label could be (2,2,2).
    assert max_len_found == 3, f"Expected max order of final polynomial to be 3, found {max_len_found}"


    # --- Verification 2: Specific coefficient value ---
    # R test: `expect_equal(results_r[[4]]$values[which(results_r[[4]]$labels == "x1x1"), 1], 0.63351833, ...)`
    # results_r[[4]] is H_2 (output of non-linear step of 2nd R layer / layer_idx=1 in Python).
    # Log indices: P0 (idx 0), Z1 (idx 1), H1 (idx 2), Z2 (idx 3), H2 (idx 4)
    log_entry_H2 = result_log[4]
    assert log_entry_H2['description'] == 'Layer 2 - Output of Non-linear Step (H_2)'
    
    labels_H2 = log_entry_H2['labels']
    values_H2 = log_entry_H2['values'] # Shape: (terms, neurons)

    # Target label: "x1x1" in R -> (1,1) in Python 1-based tuple notation
    target_label_tuple = (1,1)
    try:
        label_idx = labels_H2.index(target_label_tuple)
    except ValueError:
        pytest.fail(f"Label {target_label_tuple} not found in H2 labels: {labels_H2}")

    # Target neuron: 1st neuron in R -> neuron index 0 in Python
    neuron_idx = 0
    
    coefficient_actual = values_H2[label_idx, neuron_idx]
    coefficient_expected = 0.63351833
    
    # Using np.isclose for float comparison with tolerance
    # Default tolerance for np.isclose: rtol=1e-05, atol=1e-08
    assert np.isclose(coefficient_actual, coefficient_expected), \
        f"Coefficient mismatch for label {target_label_tuple}, neuron {neuron_idx} in H2 (Layer 2 output). " \
        f"Actual: {coefficient_actual}, Expected: {coefficient_expected}"

def test_nn2poly_list_input_default_options():
    """
    Corresponds to R test 'nn2poly with list input ... default options'.
    Uses testing_helper_1 data but with R's default options:
    max_order = 2, keep_layers = FALSE, taylor_orders (scalar) = 8.
    """
    # WEIGHTS_LIST_DATA and AF_STRING_LIST_DATA are already defined globally in this file.
    # R test uses default taylor_orders = 8 (scalar)
    # R test uses default max_order = 2
    # R test uses default keep_layers = FALSE

    # Precompute partitions for this specific scenario
    # p_vars = 2 (from WEIGHTS_LIST_DATA[0])
    # For taylor_orders_param=8 (scalar), af_list=["softplus", "softplus", "linear"]
    # processed_taylor_orders = [8, 8, 1]
    # q_max_final_order = min(prod([8,8,1]), max_order=2) = min(64, 2) = 2
    p_vars_test = WEIGHTS_LIST_DATA[0].shape[0] - 1
    taylor_orders_for_this_test = helpers.obtain_taylor_vector(8, AF_STRING_LIST_DATA)
    q_max_for_this_test = helpers.obtain_final_poly_order(2, taylor_orders_for_this_test) # Should be 2

    partitions_for_this_test = helpers.obtain_partitions_with_labels(
        p_variables=p_vars_test,
        q_max_poly_order=q_max_for_this_test,
        combinatorics_module=combinatorics,
        algorithms_module=algorithms
    )

    result = core.nn2poly(
        weights_list=WEIGHTS_LIST_DATA,
        af_string_list=AF_STRING_LIST_DATA,
        max_order=2,
        keep_layers=False,
        taylor_orders_param=8, # R default scalar
        precomputed_partitions=partitions_for_this_test,
        variable_names_list=None,
        verbose=False
    )

    # Assertion 1: Result is a single dict (keep_layers=False)
    assert isinstance(result, dict)
    assert 'labels' in result and 'values' in result
    assert not any(key.startswith('layer_') for key in result.keys()) # Not a nested dict

    # Assertion 2: Order of the last term in the final polynomial is 2
    final_labels = result['labels']
    assert isinstance(final_labels, list)
    
    max_len_found = 0
    if final_labels:
        # Assuming labels are sorted by length, then lexicographically by core.nn2poly
        max_len_found = len(final_labels[-1]) 
    
    assert max_len_found == 2, f"Expected max order of final polynomial to be 2, found {max_len_found}"

    # Assertion 3: Specific coefficient value
    # R test: `expect_equal(object$values[which(object$labels == "x1x1"),1], -4.429, tolerance = test_precision_less)`
    # R label "x1x1" -> Python label (1,1)
    # R values are terms x neurons. Python result (keep_layers=False) is also terms x neurons.
    # R `object$values[,1]` means first neuron (index 0).
    
    target_label_tuple = (1,1)
    try:
        label_idx = final_labels.index(target_label_tuple)
    except ValueError:
        pytest.fail(f"Label {target_label_tuple} not found in final labels: {final_labels}")
    
    # Assuming single output neuron from WEIGHTS_LIST_DATA structure (last W is (3,1))
    neuron_idx = 0 
    coefficient_actual = result['values'][label_idx, neuron_idx]
    coefficient_expected = -4.429 # From R test
    
    assert np.isclose(coefficient_actual, coefficient_expected, atol=1e-3), \
        f"Coefficient mismatch for label {target_label_tuple}, neuron {neuron_idx}. " \
        f"Actual: {coefficient_actual}, Expected: {coefficient_expected}"


def test_nn2poly_error_wrong_dimensions():
    """
    Corresponds to R test 'nn2poly error when wrong dimensions are given in weights'.
    Relies on check_weights_dimensions called by core.nn2poly.
    """
    # AF_STRING_LIST_DATA is already defined globally.
    # Make a copy of WEIGHTS_LIST_DATA to modify it for this test
    malformed_weights_list = [np.copy(w) for w in WEIGHTS_LIST_DATA]
    
    # Modify weights_list_data[1] (second weight matrix) to have incorrect dimensions
    # Original W2 is np.ones((3,2)). Previous W1 is np.ones((3,2)).
    # check_weights_dimensions expects W_curr.shape[0] == W_prev.shape[1] + 1
    # So, malformed_weights_list[1].shape[0] should not be WEIGHTS_LIST_DATA[0].shape[1] + 1
    # WEIGHTS_LIST_DATA[0].shape[1] is 2. So, expected input dim for W2 is 2+1=3.
    # If malformed_weights_list[1] is (4,2), it's an error.
    malformed_weights_list[1] = np.ones((4,2)) 

    with pytest.raises(ValueError, match="Weight dimensions are not compatible."):
        core.nn2poly(
            weights_list=malformed_weights_list,
            af_string_list=AF_STRING_LIST_DATA,
            max_order=MAX_ORDER_DATA, # From global scope, or set specific for this test
            taylor_orders_param=TAYLOR_ORDERS_DATA # From global scope
            # precomputed_partitions not strictly needed if it fails before that
        )

```
