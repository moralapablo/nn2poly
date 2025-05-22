import pytest
import numpy as np
from nn2poly_py import activations

# Test inputs for activation functions
TEST_INPUT_ARRAY = np.array([-10.0, -1.0, 0.0, 1.0, 10.0, 0.5, -0.5])

# === 1. Test Correctness of Activation Function Implementations ===

def test_linear_output():
    """Test the linear activation function."""
    x = TEST_INPUT_ARRAY
    expected_linear = x
    actual_linear = activations.linear(x)
    np.testing.assert_allclose(actual_linear, expected_linear, rtol=1e-7)

def test_sigmoid_output():
    """Test the sigmoid activation function."""
    x = TEST_INPUT_ARRAY
    expected_sigmoid = 1 / (1 + np.exp(-x))
    actual_sigmoid = activations.sigmoid(x)
    np.testing.assert_allclose(actual_sigmoid, expected_sigmoid, rtol=1e-7)

def test_tanh_output():
    """Test the hyperbolic tangent (tanh) activation function."""
    x = TEST_INPUT_ARRAY
    expected_tanh = np.tanh(x)
    actual_tanh = activations.tanh(x)
    np.testing.assert_allclose(actual_tanh, expected_tanh, rtol=1e-7)

def test_softplus_output():
    """Test the softplus activation function."""
    x = TEST_INPUT_ARRAY
    expected_softplus = np.log(1 + np.exp(x))
    actual_softplus = activations.softplus(x)
    np.testing.assert_allclose(actual_softplus, expected_softplus, rtol=1e-7)

# === 2. Test get_taylor_coefficients Function ===

def test_get_taylor_coefficients_unknown_function():
    """Test that get_taylor_coefficients raises an error for an unknown function."""
    with pytest.raises(ValueError, match="Unknown activation function: non_existent_function"):
        activations.get_taylor_coefficients("non_existent_function", order=2)
    # The prompt suggested (ValueError, NotImplementedError), but my current code only raises ValueError.

def test_get_taylor_coefficients_invalid_order():
    """Test that get_taylor_coefficients raises an error for invalid order."""
    with pytest.raises(ValueError, match="Order must be a non-negative integer."):
        activations.get_taylor_coefficients("linear", order=-1)
    with pytest.raises(ValueError, match="Order must be a non-negative integer."):
        activations.get_taylor_coefficients("linear", order=2.5) # type error

def test_get_taylor_coefficients_known_series():
    """Test get_taylor_coefficients for known Taylor series expansions."""

    # For 'linear', order=2, centered at 0.0
    # f(x) = x. f(0)=0, f'(0)=1, f''(0)=0. Coeffs: c0=0, c1=1/1!=1, c2=0/2!=0
    expected_linear_o2_a0 = np.array([0.0, 1.0, 0.0])
    actual_linear_o2_a0 = activations.get_taylor_coefficients('linear', order=2, around=0.0)
    np.testing.assert_allclose(actual_linear_o2_a0, expected_linear_o2_a0, rtol=1e-7, atol=1e-9)

    # For 'linear', order=1, centered at 1.0
    # f(x) = x. f(1)=1, f'(1)=1. Coeffs: c0=1, c1=1/1!=1
    expected_linear_o1_a1 = np.array([1.0, 1.0])
    actual_linear_o1_a1 = activations.get_taylor_coefficients('linear', order=1, around=1.0)
    np.testing.assert_allclose(actual_linear_o1_a1, expected_linear_o1_a1, rtol=1e-7, atol=1e-9)
    
    # For 'linear', order=3, centered at 2.0
    # f(x) = x. f(2)=2, f'(2)=1, f''(2)=0, f'''(2)=0
    # Coeffs: c0=2, c1=1, c2=0, c3=0
    expected_linear_o3_a2 = np.array([2.0, 1.0, 0.0, 0.0])
    actual_linear_o3_a2 = activations.get_taylor_coefficients('linear', order=3, around=2.0)
    np.testing.assert_allclose(actual_linear_o3_a2, expected_linear_o3_a2, rtol=1e-7, atol=1e-9)


    # For 'tanh', order=3, centered at 0.0
    # tanh(x) = x - x^3/3 + ...
    # c0=0, c1=1, c2=0, c3=-1/3
    expected_tanh_o3_a0 = np.array([0.0, 1.0, 0.0, -1.0/3.0])
    actual_tanh_o3_a0 = activations.get_taylor_coefficients('tanh', order=3, around=0.0)
    np.testing.assert_allclose(actual_tanh_o3_a0, expected_tanh_o3_a0, rtol=1e-7, atol=1e-9)

    # For 'sigmoid', order=3, centered at 0.0
    # sigmoid(x) = 1/2 + x/4 - x^3/48 + ...
    # c0=0.5, c1=0.25, c2=0, c3=-1/48
    expected_sigmoid_o3_a0 = np.array([0.5, 0.25, 0.0, -1.0/48.0])
    actual_sigmoid_o3_a0 = activations.get_taylor_coefficients('sigmoid', order=3, around=0.0)
    np.testing.assert_allclose(actual_sigmoid_o3_a0, expected_sigmoid_o3_a0, rtol=1e-7, atol=1e-9)

    # For 'softplus', order=2, centered at 0.0
    # softplus(x) = log(1+e^x). f(0)=log(2). f'(x)=e^x/(1+e^x), f'(0)=1/2.
    # f''(x)=e^x/((1+e^x)^2), f''(0)=1/4.
    # Coeffs: c0=log(2), c1=(1/2)/1! = 0.5, c2=(1/4)/2! = 0.125
    expected_softplus_o2_a0 = np.array([np.log(2.0), 0.5, 0.125])
    actual_softplus_o2_a0 = activations.get_taylor_coefficients('softplus', order=2, around=0.0)
    np.testing.assert_allclose(actual_softplus_o2_a0, expected_softplus_o2_a0, rtol=1e-7, atol=1e-9)

def test_get_taylor_coefficients_order_0():
    """Test Taylor coefficients for order 0."""
    # For 'tanh', order=0, centered at 0.0 -> f(0) = 0
    expected_tanh_o0_a0 = np.array([0.0])
    actual_tanh_o0_a0 = activations.get_taylor_coefficients('tanh', order=0, around=0.0)
    np.testing.assert_allclose(actual_tanh_o0_a0, expected_tanh_o0_a0, rtol=1e-7, atol=1e-9)

    # For 'sigmoid', order=0, centered at 0.0 -> f(0) = 0.5
    expected_sigmoid_o0_a0 = np.array([0.5])
    actual_sigmoid_o0_a0 = activations.get_taylor_coefficients('sigmoid', order=0, around=0.0)
    np.testing.assert_allclose(actual_sigmoid_o0_a0, expected_sigmoid_o0_a0, rtol=1e-7, atol=1e-9)

    # For 'linear', order=0, centered at 5.0 -> f(5.0) = 5.0
    expected_linear_o0_a5 = np.array([5.0])
    actual_linear_o0_a5 = activations.get_taylor_coefficients('linear', order=0, around=5.0)
    np.testing.assert_allclose(actual_linear_o0_a5, expected_linear_o0_a5, rtol=1e-7, atol=1e-9)

```
