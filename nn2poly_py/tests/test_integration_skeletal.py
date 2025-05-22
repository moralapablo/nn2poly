import pytest
import numpy as np
from nn2poly_py import core, helpers, combinatorics, algorithms # For precomputed_partitions

# === Skeletal Test for Keras-like Data ===

# Define Model Structure (Python equivalent of keras_test_model()):
# Layer 1: 2 inputs, 2 neurons, 'tanh'
# Layer 2: 2 inputs, 3 neurons, 'softplus' (R test used 'relu', mapped to 'softplus')
# Layer 3: 3 inputs, 2 neurons, 'linear'
KERAS_LIKE_AF_LIST = ['tanh', 'softplus', 'linear']

# Manually define simple, non-random weights that fit the Keras architecture.
# These are for structural testing, not for matching R's random weight values.
# Shapes: W_l is (inputs_inc_bias, outputs)
# L1: Input_dim=2, Units=2. W_shape=(2+1, 2) = (3,2)
# L2: Input_dim=2 (from L1 units), Units=3. W_shape=(2+1, 3) = (3,3)
# L3: Input_dim=3 (from L2 units), Units=2. W_shape=(3+1, 2) = (4,2)
KERAS_LIKE_WEIGHTS_LIST = [
    np.array([[0.1, 0.2],        # Bias for L1 neurons
              [1.0, -1.0],       # Kernel for L1, input 1
              [0.5, 0.5]]),      # Kernel for L1, input 2
    
    np.array([[0.3, 0.4, 0.5],  # Bias for L2 neurons
              [1.0, 0.0, -1.0],  # Kernel for L2, input 1 (from L1 neuron 1)
              [0.0, 1.0, 0.0]]), # Kernel for L2, input 2 (from L1 neuron 2)
              
    np.array([[0.6, 0.7],        # Bias for L3 neurons
              [1.0, 0.0],        # Kernel for L3, input 1 (from L2 neuron 1)
              [0.0, 1.0],        # Kernel for L3, input 2 (from L2 neuron 2)
              [0.5, 0.5]])       # Kernel for L3, input 3 (from L2 neuron 3)
]

# R test uses taylor_orders = c(2,2,1) and max_order = 2
KERAS_LIKE_TAYLOR_ORDERS = np.array([2,2,1])
KERAS_LIKE_MAX_ORDER = 2

def test_nn2poly_with_keras_like_data():
    """
    Skeletal test using data shaped like Keras model output.
    Focuses on structural correctness and ensuring nn2poly runs.
    Exact numerical matches for coefficients are not asserted due to
    differences in weight initialization from R's Keras environment.
    """
    p_vars_keras = KERAS_LIKE_WEIGHTS_LIST[0].shape[0] - 1 # Should be 2
    
    # Calculate q_max_final_order based on test parameters
    # processed_taylor_orders for ['tanh', 'softplus', 'linear'] with [2,2,1] is [2,2,1]
    # q_max_final_order = min(prod([2,2,1]), max_order=2) = min(4,2) = 2
    q_max_for_keras_test = helpers.obtain_final_poly_order(
        KERAS_LIKE_MAX_ORDER,
        helpers.obtain_taylor_vector(KERAS_LIKE_TAYLOR_ORDERS, KERAS_LIKE_AF_LIST)
    )
    assert q_max_for_keras_test == 2

    keras_partitions = helpers.obtain_partitions_with_labels(
        p_variables=p_vars_keras,
        q_max_poly_order=q_max_for_keras_test,
        combinatorics_module=combinatorics,
        algorithms_module=algorithms
    )

    result = core.nn2poly(
        weights_list=KERAS_LIKE_WEIGHTS_LIST,
        af_string_list=KERAS_LIKE_AF_LIST,
        max_order=KERAS_LIKE_MAX_ORDER,
        keep_layers=False, # R test checks final poly
        taylor_orders_param=KERAS_LIKE_TAYLOR_ORDERS,
        precomputed_partitions=keras_partitions
    )

    # Assertions from R test (these will likely fail with placeholder weights,
    # but are included to show the structure of what R was testing):
    # R: `expect_equal(result$values[1,1],  0.18148204)`
    # R: `expect_equal(result$values[3,2], -0.71466625)`
    # R: `expect_equal(result$labels[[6]], c(2,2))`

    # Python structural assertions:
    assert isinstance(result, dict)
    assert 'labels' in result and 'values' in result
    assert isinstance(result['labels'], list)
    assert isinstance(result['values'], np.ndarray)
    assert result['values'].ndim == 2 # terms x neurons

    # Check a specific label based on max_order and p_vars
    # For max_order=2, p_vars=2, labels could be (), (1), (2), (1,1), (1,2), (2,2)
    # The R test expects labels to be sorted and result$labels[[6]] is c(2,2) (1-based)
    # Python equivalent (1-based for content, 0-based for list index):
    # Expected sorted labels (Python 1-based tuples):
    # (), (1,), (2,), (1,1), (1,2), (2,2) -> 6 labels total
    # So, result['labels'][5] should be (2,2)
    
    # First, check if the number of labels is as expected for p_vars=2, max_order=2
    # Number of terms = sum_{k=0 to Q} choose(P+k-1, k)
    # Q=2, P=2:
    # k=0: choose(2-1,0) = choose(1,0) = 1 (intercept)
    # k=1: choose(2+1-1,1) = choose(2,1) = 2 (x1, x2)
    # k=2: choose(2+2-1,2) = choose(3,2) = 3 (x1x1, x1x2, x2x2)
    # Total = 1 + 2 + 3 = 6 labels.
    assert len(result['labels']) == 6, "Number of labels does not match expected for p=2, Q=2."
    
    # Check the last label (highest order, lexicographically last)
    # My core.nn2poly sorts labels: by length, then lexicographically
    expected_labels_sorted = [tuple(), (1,), (2,), (1,1), (1,2), (2,2)]
    assert result['labels'] == expected_labels_sorted
    assert result['labels'][5] == (2,2), "Expected 6th label to be (2,2)"

    # Check dimensions of values array: (num_labels, num_output_neurons_of_last_layer)
    # Last layer (L3) has 2 output neurons.
    assert result['values'].shape == (6, 2), "Shape of values array is not (num_labels, num_L3_neurons)"
    
    # We cannot assert specific coefficient values without matching R's random seed & Keras version.
    # print(f"Keras-like test: result['values'][0,0] = {result['values'][0,0]}") # Coeff of intercept for 1st neuron
    # print(f"Keras-like test: result['values'][result['labels'].index((2,2)), 1] = {result['values'][result['labels'].index((2,2)), 1]}") # Coeff of (2,2) for 2nd neuron

# === Skeletal Test for PyTorch-like Data ===

# Define Model Structure (Python equivalent of luz_test_model()):
# Linear(2,2) -> Softplus
# Linear(2,3) -> Softplus
# Linear(3,1) -> Linear
TORCH_LIKE_AF_LIST = ['softplus', 'softplus', 'linear']

# Manually define simple, non-random weights.
# L1: In=2, Out=2. W_shape=(2+1, 2) = (3,2)
# L2: In=2 (from L1 units), Out=3. W_shape=(2+1, 3) = (3,3)
# L3: In=3 (from L2 units), Out=1. W_shape=(3+1, 1) = (4,1)
TORCH_LIKE_WEIGHTS_LIST = [
    np.array([[0.1, 0.2], [1.0, -1.0], [0.5, 0.5]]),
    np.array([[0.3, 0.4, 0.5], [1.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
    np.array([[0.6], [1.0], [0.0], [0.5]]) # Bias(1,1), Kernel(3,1)
]

# R test uses testing_data$taylor_orders = c(2,2,1) and max_order = 3
TORCH_LIKE_TAYLOR_ORDERS = np.array([2,2,1])
TORCH_LIKE_MAX_ORDER = 3

def test_nn2poly_with_pytorch_like_data():
    """
    Skeletal test using data shaped like PyTorch model output.
    Focuses on structural correctness. Numerical value checks are illustrative.
    """
    p_vars_torch = TORCH_LIKE_WEIGHTS_LIST[0].shape[0] - 1 # Should be 2

    # Calculate q_max_final_order based on test parameters
    # processed_taylor_orders for ['softplus', 'softplus', 'linear'] with [2,2,1] is [2,2,1]
    # q_max_final_order = min(prod([2,2,1]), max_order=3) = min(4,3) = 3
    q_max_for_torch_test = helpers.obtain_final_poly_order(
        TORCH_LIKE_MAX_ORDER,
        helpers.obtain_taylor_vector(TORCH_LIKE_TAYLOR_ORDERS, TORCH_LIKE_AF_LIST)
    )
    assert q_max_for_torch_test == 3

    torch_partitions = helpers.obtain_partitions_with_labels(
        p_variables=p_vars_torch,
        q_max_poly_order=q_max_for_torch_test,
        combinatorics_module=combinatorics,
        algorithms_module=algorithms
    )

    result = core.nn2poly(
        weights_list=TORCH_LIKE_WEIGHTS_LIST,
        af_string_list=TORCH_LIKE_AF_LIST,
        max_order=TORCH_LIKE_MAX_ORDER,
        keep_layers=False,
        taylor_orders_param=TORCH_LIKE_TAYLOR_ORDERS,
        precomputed_partitions=torch_partitions
    )

    # Assertions from R test (numerical values are illustrative):
    # R: `expect_equal(round(result$values[1,1],2), 0.06)`
    # R: `expect_equal(result$labels[[7]], c(1,1,1))`

    assert isinstance(result, dict)
    assert 'labels' in result and 'values' in result
    assert isinstance(result['labels'], list)
    assert isinstance(result['values'], np.ndarray)
    assert result['values'].ndim == 2 # terms x neurons

    # Check structure for p_vars=2, max_order=3
    # Number of terms = sum_{k=0 to Q} choose(P+k-1, k)
    # Q=3, P=2:
    # k=0: 1 (intercept)
    # k=1: 2 (x1, x2)
    # k=2: 3 (x1x1, x1x2, x2x2)
    # k=3: choose(2+3-1,3) = choose(4,3) = 4 (x1x1x1, x1x1x2, x1x2x2, x2x2x2)
    # Total = 1 + 2 + 3 + 4 = 10 labels.
    assert len(result['labels']) == 10, "Number of labels does not match expected for p=2, Q=3."

    # Check the 7th label (index 6)
    # Expected sorted labels (Python 1-based tuples):
    # (), (1,), (2,), (1,1), (1,2), (2,2), (1,1,1), (1,1,2), (1,2,2), (2,2,2)
    expected_labels_sorted_torch = [
        tuple(), (1,), (2,), (1,1), (1,2), (2,2), 
        (1,1,1), (1,1,2), (1,2,2), (2,2,2)
    ]
    assert result['labels'] == expected_labels_sorted_torch
    assert result['labels'][6] == (1,1,1), "Expected 7th label to be (1,1,1)"

    # Check dimensions of values array: (num_labels, num_output_neurons_of_last_layer)
    # Last layer (L3) has 1 output neuron.
    assert result['values'].shape == (10, 1), "Shape of values array is not (num_labels, num_L3_neurons)"

    # Illustrative check for a coefficient (will not match R's random value)
    # print(f"Torch-like test: Coeff for intercept: {result['values'][0,0]}") # R: round(result$values[1,1],2), 0.06
    # This would be result['values'][result['labels'].index(tuple()), 0]
    # For example, if we set all weights and biases to simple values (e.g., 0 or 1),
    # we could manually calculate an expected coefficient.
    # With the current simple weights, the values will be some specific numbers.
    # For example, intercept coeff: result['values'][0,0]
    # This is the coefficient of the constant term for the single output neuron.
    # It's hard to manually verify without stepping through the full nn2poly logic.
    # The structural checks are more reliable for these skeletal tests.
```
