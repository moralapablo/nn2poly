import pytest
import numpy as np
import pandas as pd
from nn2poly_py import polynomial, core, helpers, combinatorics, algorithms # For full nn2poly run

# === Test Data Setup (adapted from test_polynomial.py and R tests) ===

# Single Polynomial (POLY_A: like POLY1 from test_polynomial.py)
# Represents output of nn2poly(..., keep_layers=False)
# Or a component (e.g., layer_1$input) from keep_layers=True
POLY_A_LABELS = [(0,), (1,), (2,), (1,1)] # Intercept, x1, x2, x1^2
POLY_A_VALUES = np.array([1, -1, 1, 0.5]).reshape(-1, 1) # (4 terms, 1 poly/neuron)
POLY_A = {'labels': POLY_A_LABELS, 'values': POLY_A_VALUES}

# Another Single Polynomial (POLY_B: like a column from POLY2)
POLY_B_LABELS = [(0,), (1,), (2,)] # Intercept, x1, x2
POLY_B_VALUES = np.array([2, 3, -2]).reshape(-1, 1) # (3 terms, 1 poly/neuron)
POLY_B = {'labels': POLY_B_LABELS, 'values': POLY_B_VALUES}

# Polynomial with multiple output neurons (POLY_C: like POLY2 from test_polynomial.py)
POLY_C_LABELS = [(0,), (1,), (2,)]
POLY_C_VALUES = np.array([[1,2], [-1,3], [1,-2]]) # (3 terms, 2 polys/neurons)
POLY_C = {'labels': POLY_C_LABELS, 'values': POLY_C_VALUES}


# Newdata
NEWDATA_SINGLE_OBS_2_FEATURES = np.array([[1, 1]]) # R: c(1,1)
NEWDATA_MULTI_OBS_2_FEATURES = np.array([[1, 2], [1, 1]]) # R: rbind(c(1,2), c(1,1))

# === Tests for predict_nn2poly with Single Polynomial Input ===

def test_predict_single_poly_single_obs():
    # Expected: 1 - 1 + 1 + 0.5*1^2 = 1.5
    expected = 1.5 
    actual = polynomial.predict_nn2poly(POLY_A, NEWDATA_SINGLE_OBS_2_FEATURES)
    assert isinstance(actual, float) # eval_poly squeezes for single obs, single poly
    np.testing.assert_allclose(actual, expected)

def test_predict_monomials_sum_equals_poly_eval_single_poly():
    # Check consistency: sum(predict(monomials=True)) == predict(monomials=False)
    pred_monomials = polynomial.predict_nn2poly(POLY_A, NEWDATA_SINGLE_OBS_2_FEATURES, monomials=True)
    pred_summed = polynomial.predict_nn2poly(POLY_A, NEWDATA_SINGLE_OBS_2_FEATURES, monomials=False)
    
    # pred_monomials for single poly, single obs is 1D (n_terms,)
    # pred_summed is scalar
    assert pred_monomials.ndim == 1
    np.testing.assert_allclose(np.sum(pred_monomials), pred_summed)

    # Test with multiple observations
    pred_monomials_multi_obs = polynomial.predict_nn2poly(POLY_A, NEWDATA_MULTI_OBS_2_FEATURES, monomials=True)
    pred_summed_multi_obs = polynomial.predict_nn2poly(POLY_A, NEWDATA_MULTI_OBS_2_FEATURES, monomials=False)
    # pred_monomials_multi_obs for single poly, multi obs is 2D (n_obs, n_terms)
    # pred_summed_multi_obs is 1D (n_obs,)
    assert pred_monomials_multi_obs.ndim == 2
    np.testing.assert_allclose(np.sum(pred_monomials_multi_obs, axis=1), pred_summed_multi_obs)

def test_predict_multiple_poly_single_obs_as_single_poly_dict():
    # POLY_C has 2 output polynomials/neurons
    # Newdata: c(1,1) -> x1=1, x2=1
    # Poly1: 1 - 1 + 1 = 1
    # Poly2: 2 + 3 - 2 = 3
    # Expected: array([1,3])
    expected = np.array([1,3])
    actual = polynomial.predict_nn2poly(POLY_C, NEWDATA_SINGLE_OBS_2_FEATURES)
    assert actual.ndim == 1 # eval_poly squeezes for single obs, multi poly
    np.testing.assert_allclose(actual, expected)

def test_predict_dataframe_input_single_poly():
    df_single_obs = pd.DataFrame({'X1': [1], 'X2': [1]}) # Column names don't matter for eval_poly
    expected = 1.5
    actual = polynomial.predict_nn2poly(POLY_A, df_single_obs)
    assert isinstance(actual, float)
    np.testing.assert_allclose(actual, expected)

# === Tests for predict_nn2poly with Multi-Layer (Nested Dict) Input ===

# Sample multi-layer object
# Using POLY_A and POLY_B for simplicity. In reality, 'input' and 'output' polys
# for a layer would have different structures/coefficients.
# Values are (terms, neurons)
TEST_OBJ_MULTI = {
    'layer_1': {
        'input': POLY_A,    # Simulate Z_1 (output of linear step of layer 1)
        'output': POLY_B     # Simulate H_1 (output of non-linear step of layer 1)
    },
    'layer_2': {
        'input': POLY_B,    # Simulate Z_2 (output of linear step of layer 2)
        'output': POLY_A     # Simulate H_2 (output of non-linear step of layer 2)
    }
}

def test_predict_multiple_layers_eval_on_each_component():
    predictions = polynomial.predict_nn2poly(TEST_OBJ_MULTI, NEWDATA_SINGLE_OBS_2_FEATURES)
    
    assert 'layer_1' in predictions
    assert 'layer_2' in predictions
    assert 'input' in predictions['layer_1']
    assert 'output' in predictions['layer_1']

    # Compare with direct eval_poly calls
    expected_l1_input = polynomial.eval_poly(TEST_OBJ_MULTI['layer_1']['input'], NEWDATA_SINGLE_OBS_2_FEATURES)
    expected_l1_output = polynomial.eval_poly(TEST_OBJ_MULTI['layer_1']['output'], NEWDATA_SINGLE_OBS_2_FEATURES)
    expected_l2_input = polynomial.eval_poly(TEST_OBJ_MULTI['layer_2']['input'], NEWDATA_SINGLE_OBS_2_FEATURES)
    expected_l2_output = polynomial.eval_poly(TEST_OBJ_MULTI['layer_2']['output'], NEWDATA_SINGLE_OBS_2_FEATURES)

    np.testing.assert_allclose(predictions['layer_1']['input'], expected_l1_input)
    np.testing.assert_allclose(predictions['layer_1']['output'], expected_l1_output)
    np.testing.assert_allclose(predictions['layer_2']['input'], expected_l2_input)
    np.testing.assert_allclose(predictions['layer_2']['output'], expected_l2_output)

    # Test with monomials=True
    predictions_m_true = polynomial.predict_nn2poly(TEST_OBJ_MULTI, NEWDATA_SINGLE_OBS_2_FEATURES, monomials=True)
    expected_l1_input_m_true = polynomial.eval_poly(TEST_OBJ_MULTI['layer_1']['input'], NEWDATA_SINGLE_OBS_2_FEATURES, monomials=True)
    np.testing.assert_allclose(predictions_m_true['layer_1']['input'], expected_l1_input_m_true)


def test_predict_multiple_layers_select_layer():
    # Select layer 2
    predictions_l2 = polynomial.predict_nn2poly(TEST_OBJ_MULTI, NEWDATA_SINGLE_OBS_2_FEATURES, layers=2)
    assert 'layer_2' in predictions_l2
    assert 'layer_1' not in predictions_l2
    assert len(predictions_l2) == 1

    expected_l2_output = polynomial.eval_poly(TEST_OBJ_MULTI['layer_2']['output'], NEWDATA_SINGLE_OBS_2_FEATURES)
    np.testing.assert_allclose(predictions_l2['layer_2']['output'], expected_l2_output)

    # Select layer 1 using a list
    predictions_l1_list = polynomial.predict_nn2poly(TEST_OBJ_MULTI, NEWDATA_SINGLE_OBS_2_FEATURES, layers=[1])
    assert 'layer_1' in predictions_l1_list
    assert 'layer_2' not in predictions_l1_list
    assert len(predictions_l1_list) == 1
    expected_l1_input = polynomial.eval_poly(TEST_OBJ_MULTI['layer_1']['input'], NEWDATA_SINGLE_OBS_2_FEATURES)
    np.testing.assert_allclose(predictions_l1_list['layer_1']['input'], expected_l1_input)


def test_predict_multiple_layers_select_layers_list():
    predictions_multi = polynomial.predict_nn2poly(TEST_OBJ_MULTI, NEWDATA_MULTI_OBS_2_FEATURES, layers=[1, 2])
    assert 'layer_1' in predictions_multi
    assert 'layer_2' in predictions_multi
    assert len(predictions_multi) == 2

    expected_l1_output = polynomial.eval_poly(TEST_OBJ_MULTI['layer_1']['output'], NEWDATA_MULTI_OBS_2_FEATURES)
    np.testing.assert_allclose(predictions_multi['layer_1']['output'], expected_l1_output)

    expected_l2_input = polynomial.eval_poly(TEST_OBJ_MULTI['layer_2']['input'], NEWDATA_MULTI_OBS_2_FEATURES)
    np.testing.assert_allclose(predictions_multi['layer_2']['input'], expected_l2_input)
    
    # Test with duplicate and unordered list
    predictions_dup_unordered = polynomial.predict_nn2poly(TEST_OBJ_MULTI, NEWDATA_MULTI_OBS_2_FEATURES, layers=[2, 1, 2])
    assert len(predictions_dup_unordered) == 2 # Should be unique and sorted by key
    assert list(predictions_dup_unordered.keys()) == ['layer_1', 'layer_2'] # Check sorted order


def test_predict_layers_argument_errors():
    with pytest.raises(ValueError, match="layers argument must be None, an integer, or a list of unique integers."):
        polynomial.predict_nn2poly(TEST_OBJ_MULTI, NEWDATA_SINGLE_OBS_2_FEATURES, layers="2")
    
    with pytest.raises(ValueError, match="layers argument must be None, an integer, or a list of unique integers."):
        polynomial.predict_nn2poly(TEST_OBJ_MULTI, NEWDATA_SINGLE_OBS_2_FEATURES, layers=[1, "2"])

    # Test layer not found (KeyError if specific layers requested and none found)
    with pytest.raises(KeyError, match="None of the requested layers .* found"):
        polynomial.predict_nn2poly(TEST_OBJ_MULTI, NEWDATA_SINGLE_OBS_2_FEATURES, layers=3)
        
    with pytest.warns(UserWarning, match="Layer layer_3 not found"):
        result = polynomial.predict_nn2poly(TEST_OBJ_MULTI, NEWDATA_SINGLE_OBS_2_FEATURES, layers=[1,3])
        assert 'layer_1' in result
        assert 'layer_3' not in result # Only existing layers are returned

# === Integration Test with core.nn2poly output ===
# Using data from test_core.py
WEIGHTS_LIST_DATA_CORE = [
    np.ones((3, 2)),  # Layer 1: W_shape (inputs_inc_bias, outputs) = (2+1, 2)
    np.ones((3, 2)),  # Layer 2: W_shape (2+1, 2)
    np.ones((3, 1))   # Layer 3: W_shape (2+1, 1)
]
AF_STRING_LIST_DATA_CORE = ["softplus", "softplus", "linear"]
TAYLOR_ORDERS_DATA_CORE = np.array([2, 2, 1])
MAX_ORDER_DATA_CORE = 3

# Precompute partitions for the core_nn2poly_object
p_vars_core = WEIGHTS_LIST_DATA_CORE[0].shape[0] - 1 # Should be 2
q_max_eff_core = helpers.obtain_final_poly_order(MAX_ORDER_DATA_CORE, TAYLOR_ORDERS_DATA_CORE) # Should be 3
PRECOMPUTED_PARTITIONS_CORE = helpers.obtain_partitions_with_labels(
    p_vars_core, q_max_eff_core, combinatorics, algorithms
)

# Generate the multi-layer object using core.nn2poly
NN2POLY_OUTPUT_KEPT_LAYERS = core.nn2poly(
    weights_list=WEIGHTS_LIST_DATA_CORE,
    af_string_list=AF_STRING_LIST_DATA_CORE,
    max_order=MAX_ORDER_DATA_CORE,
    keep_layers=True,
    taylor_orders_param=TAYLOR_ORDERS_DATA_CORE,
    precomputed_partitions=PRECOMPUTED_PARTITIONS_CORE
)

def test_predict_with_actual_nn2poly_output():
    newdata = NEWDATA_MULTI_OBS_2_FEATURES # Use (2 obs, 2 features)

    # Predict for all layers
    pred_all = polynomial.predict_nn2poly(NN2POLY_OUTPUT_KEPT_LAYERS, newdata)
    assert len(pred_all) == len(AF_STRING_LIST_DATA_CORE) # Should have predictions for 3 layers
    assert 'layer_1' in pred_all and 'layer_3' in pred_all

    # Compare layer 1 output from predict_nn2poly with direct eval_poly
    expected_l1_output = polynomial.eval_poly(NN2POLY_OUTPUT_KEPT_LAYERS['layer_1']['output'], newdata)
    np.testing.assert_allclose(pred_all['layer_1']['output'], expected_l1_output)

    # Predict for a specific layer (e.g., layer 2)
    pred_l2 = polynomial.predict_nn2poly(NN2POLY_OUTPUT_KEPT_LAYERS, newdata, layers=2)
    assert len(pred_l2) == 1
    assert 'layer_2' in pred_l2
    assert 'layer_1' not in pred_l2

    # Compare layer 2 output from pred_l2 with direct eval_poly
    expected_l2_output_direct = polynomial.eval_poly(NN2POLY_OUTPUT_KEPT_LAYERS['layer_2']['output'], newdata)
    np.testing.assert_allclose(pred_l2['layer_2']['output'], expected_l2_output_direct)
    
    # Compare layer 2 input
    expected_l2_input_direct = polynomial.eval_poly(NN2POLY_OUTPUT_KEPT_LAYERS['layer_2']['input'], newdata)
    np.testing.assert_allclose(pred_l2['layer_2']['input'], expected_l2_input_direct)

    # R test: expect_equal(prediction1$layer_2$output, prediction1$layer_2$input)
    # This specific assertion depends on the model and data.
    # For the given weights (all ones) and softplus, it's unlikely input == output for layer 2.
    # The R test might be on a different model setup or a specific case where this holds.
    # We'll skip this specific R assertion unless the data matches that expectation.
    # For this test data, Z2 (input to layer 2 activation) and H2 (output of layer 2 activation)
    # will generally not be the same if softplus (order 2) is applied.
    # If layer 2 was 'linear' with order 1, then Z2 would be very similar to H2.
    
    # Test with monomials=True for a specific layer
    pred_l3_monomials = polynomial.predict_nn2poly(NN2POLY_OUTPUT_KEPT_LAYERS, newdata, layers=3, monomials=True)
    assert 'layer_3' in pred_l3_monomials
    expected_l3_output_monomials = polynomial.eval_poly(NN2POLY_OUTPUT_KEPT_LAYERS['layer_3']['output'], newdata, monomials=True)
    np.testing.assert_allclose(pred_l3_monomials['layer_3']['output'], expected_l3_output_monomials)

```
