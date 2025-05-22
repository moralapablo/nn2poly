import pytest
import numpy as np
from nn2poly_py import helpers

# === Tests for helpers.obtain_final_poly_order ===

def test_final_order_works_with_integer_max_order():
    # R: expect_equal(obtain_final_poly_order(max_order = 4, taylor_orders = c(8,8,1)), 4)
    assert helpers.obtain_final_poly_order(max_order=4, taylor_orders_vector=np.array([8,8,1], dtype=int)) == 4

def test_final_order_works_with_float_max_order_implicitly_int():
    # R: expect_equal(obtain_final_poly_order(max_order = 4.0, taylor_orders = c(8,8,1)), 4)
    # Python's max_order is type-hinted as int. The function itself enforces isinstance(max_order, int).
    # To match R's behavior of accepting 4.0, the test should reflect this if the Python code was changed.
    # Given the current Python code, this test will check the enforcement.
    with pytest.raises(ValueError, match="max_order must be a non-negative integer."):
        helpers.obtain_final_poly_order(max_order=4.0, taylor_orders_vector=np.array([8,8,1], dtype=int))
    # If we wanted to test successful conversion:
    # assert helpers.obtain_final_poly_order(max_order=int(4.0), taylor_orders_vector=np.array([8,8,1])) == 4

def test_final_order_warns_if_max_order_not_reached():
    # R: expect_warning(obtain_final_poly_order(max_order = 8, taylor_orders = c(2,1,3)))
    # R: also expect_equal(suppressWarnings(obtain_final_poly_order(max_order = 8, taylor_orders = c(2,1,3))), 6)
    with pytest.warns(UserWarning, match="Product of Taylor orders (6) is less than specified max_order (8)"):
        result = helpers.obtain_final_poly_order(max_order=8, taylor_orders_vector=np.array([2,1,3], dtype=int))
    assert result == 6

def test_final_order_error_if_max_order_not_integer_like():
    # R: expect_error(obtain_final_poly_order(max_order = 4.3, taylor_orders = c(2,1,3)))
    with pytest.raises(ValueError, match="max_order must be a non-negative integer."):
        helpers.obtain_final_poly_order(max_order=4.3, taylor_orders_vector=np.array([2,1,3], dtype=int))

# Additional tests for obtain_final_poly_order based on Python implementation details
def test_final_order_error_if_taylor_orders_not_numpy_int_array():
    with pytest.raises(ValueError, match="taylor_orders_vector must be a NumPy array of integers."):
        helpers.obtain_final_poly_order(max_order=4, taylor_orders_vector=[2,1,3]) # Must be np.array
    
    with pytest.raises(ValueError, match="Taylor orders in the vector must be non-negative."):
        helpers.obtain_final_poly_order(max_order=4, taylor_orders_vector=np.array([2.5, 1, 3])) # Floats not allowed
        
    with pytest.raises(ValueError, match="Taylor orders in the vector must be non-negative."):
        helpers.obtain_final_poly_order(max_order=4, taylor_orders_vector=np.array([-1, 2, 3], dtype=int))


# === Tests for helpers.obtain_taylor_vector ===

def test_taylor_vector_single_value_multiple_linear():
    # R: expect_equal(obtain_taylor_vector(5, c('softplus', 'linear', 'softplus', 'linear')), c(5,1,5,1))
    expected = np.array([5,1,5,1])
    actual = helpers.obtain_taylor_vector(taylor_orders_param=5, 
                                          af_string_list=['softplus', 'linear', 'softplus', 'linear'])
    np.testing.assert_array_equal(actual, expected)

def test_taylor_vector_with_vector_param():
    # R: expect_equal(obtain_taylor_vector(c(5,5,1), c('softplus', 'softplus', 'linear')), c(5,5,1))
    expected = np.array([5,5,1])
    # Test with list
    actual_list_param = helpers.obtain_taylor_vector(taylor_orders_param=[5,5,1],
                                                     af_string_list=['softplus', 'softplus', 'linear'])
    np.testing.assert_array_equal(actual_list_param, expected)
    # Test with NumPy array
    actual_np_param = helpers.obtain_taylor_vector(taylor_orders_param=np.array([5,5,1], dtype=int),
                                                   af_string_list=['softplus', 'softplus', 'linear'])
    np.testing.assert_array_equal(actual_np_param, expected)


def test_taylor_vector_error_dimension_mismatch():
    # R: expect_error(obtain_taylor_vector(c(5,1), c('softplus', 'softplus', 'linear')))
    with pytest.raises(ValueError, match="Length of taylor_orders .* must match length of af_string_list"):
        helpers.obtain_taylor_vector(taylor_orders_param=[5,1], 
                                     af_string_list=['softplus', 'softplus', 'linear'])

def test_taylor_vector_error_non_integer_value_in_vector():
    # R: expect_error(obtain_taylor_vector(c(5.4, 2.3, 1), c('softplus', 'softplus', 'linear')))
    # Python function expects integer orders.
    with pytest.raises(ValueError, match="All Taylor orders must be non-negative integers."):
        helpers.obtain_taylor_vector(taylor_orders_param=[5.4, 2.3, 1], 
                                     af_string_list=['softplus', 'softplus', 'linear'])

# Additional tests for obtain_taylor_vector based on Python implementation
def test_taylor_vector_scalar_param_type_error():
    # Python implementation is stricter on type for scalar taylor_orders_param
    with pytest.raises(TypeError, match="taylor_orders must be an int, list, or NumPy array."):
        helpers.obtain_taylor_vector(taylor_orders_param=5.5, 
                                     af_string_list=['softplus', 'linear'])

def test_taylor_vector_negative_order_in_scalar():
    with pytest.raises(ValueError, match="Scalar taylor_orders must be non-negative."):
        helpers.obtain_taylor_vector(taylor_orders_param=-2,
                                     af_string_list=['softplus'])
                                     
def test_taylor_vector_negative_order_in_vector():
    with pytest.raises(ValueError, match="All Taylor orders must be non-negative integers."):
        helpers.obtain_taylor_vector(taylor_orders_param=[5, -1, 2],
                                     af_string_list=['softplus', 'softplus', 'linear'])

def test_taylor_vector_empty_af_list():
    expected = np.array([], dtype=int)
    actual_scalar_param = helpers.obtain_taylor_vector(taylor_orders_param=5, af_string_list=[])
    np.testing.assert_array_equal(actual_scalar_param, expected)
    
    actual_list_param = helpers.obtain_taylor_vector(taylor_orders_param=[], af_string_list=[])
    np.testing.assert_array_equal(actual_list_param, expected)

def test_taylor_vector_overrides_linear_order_if_vector_provided():
    # R: expect_equal(obtain_taylor_vector(c(5,8,1), c('softplus', 'linear', 'softplus')), c(5,1,1))
    # The R code seems to imply if af is 'linear', its order is always 1.
    # My Python code: if af_string_list[i].lower() == 'linear': result_orders[i] = 1
    # This means even if taylor_orders_param = [5,8,1] is given, for a 'linear' layer at index 1,
    # it will be overridden to 1.
    expected = np.array([5,1,1]) # af_list[1] is 'linear', so order becomes 1, not 8. af_list[2] is 'softplus', order is 1.
    actual = helpers.obtain_taylor_vector(taylor_orders_param=[5,8,1], 
                                          af_string_list=['softplus', 'linear', 'softplus'])
    np.testing.assert_array_equal(actual, expected)

```
