import pytest
import numpy as np
import pandas as pd # For testing DataFrame input
from nn2poly_py import polynomial # Assuming the file is polynomial.py

# === Test Data Setup ===

# Poly 1 (single polynomial, from R tests)
# labels = list(c(0), c(1), c(2), c(1,1))
# values = c(1, -1, 1, 0.5)  # Intercept, x1, x2, x1^2
POLY1_LABELS = [(0,), (1,), (2,), (1,1)]
POLY1_VALUES = np.array([1, -1, 1, 0.5]).reshape(-1, 1) # (4 terms, 1 poly)
POLY1 = {'labels': POLY1_LABELS, 'values': POLY1_VALUES}

# Poly 1 Unordered (same as Poly1 but labels initially out of order)
POLY1_UNORDERED_LABELS = [(1,1), (0,), (2,), (1,)]
POLY1_UNORDERED_VALUES = np.array([0.5, 1, 1, -1]).reshape(-1, 1) # Coeffs correspond to new label order
POLY1_UNORDERED = {'labels': POLY1_UNORDERED_LABELS, 'values': POLY1_UNORDERED_VALUES}


# Poly 2 (multiple polynomials, from R tests)
# labels = list(c(0),c(1),c(2))
# values = matrix(c(1,-1,1, 2,3,-2), ncol=2, byrow=FALSE)
# R matrix: 1  2
#          -1  3
#           1 -2
POLY2_LABELS = [(0,), (1,), (2,)]
POLY2_VALUES = np.array([[1,2], [-1,3], [1,-2]]) # (3 terms, 2 polys)
POLY2 = {'labels': POLY2_LABELS, 'values': POLY2_VALUES}

# Poly 2 No Intercept
# labels = list(c(1),c(2))
# values = matrix(c(-1,1, 3,-2), ncol=2, byrow=FALSE)
POLY2_NO_INTERCEPT_LABELS = [(1,), (2,)]
POLY2_NO_INTERCEPT_VALUES = np.array([[-1,3], [1,-2]]) # (2 terms, 2 polys)
POLY2_NO_INTERCEPT = {'labels': POLY2_NO_INTERCEPT_LABELS, 'values': POLY2_NO_INTERCEPT_VALUES}

# Poly 2 Unordered
# labels = list(c(2),c(0),c(1)) # x2, intercept, x1
# values = matrix(c(1,1,-1, -2,2,3), ncol=2, byrow=FALSE) # c(1,-2, 1,2, -1,3) after sorting by label
# R values (col-major):
# 1  -2  (for x2)
# 1   2  (for intercept)
#-1   3  (for x1)
# Corresponding to python labels [(2,), (0,), (1,)]:
POLY2_UNORDERED_LABELS = [(2,), (0,), (1,)]
POLY2_UNORDERED_VALUES = np.array([[1,-2], [1,2], [-1,3]])
POLY2_UNORDERED = {'labels': POLY2_UNORDERED_LABELS, 'values': POLY2_UNORDERED_VALUES}


# Poly 3 (high order first, no intercept)
# labels = list(c(1,1,1),c(1),c(2))
# values = c(1,2,3)
POLY3_LABELS = [(1,1,1), (1,), (2,)]
POLY3_VALUES = np.array([1,2,3]).reshape(-1,1)
POLY3 = {'labels': POLY3_LABELS, 'values': POLY3_VALUES}


# Newdata
NEWDATA1_SINGLE_OBS = np.array([[1, 1]]) # R: c(1,1)
NEWDATA2_SINGLE_OBS = np.array([[1, 2]]) # R: c(1,2)

NEWDATA_MULTI_OBS = np.array([[1, 2], [1, 1]]) # R: rbind(c(1,2), c(1,1))

# === Tests for monomials=False ===

def test_single_poly_single_obs():
    # R: eval_poly(poly1, c(1,1)) == 1 - 1 + 1 + 0.5*1^2 == 1.5
    expected1 = 1.5 
    actual1 = polynomial.eval_poly(POLY1, NEWDATA1_SINGLE_OBS)
    assert isinstance(actual1, float) # Expect scalar
    np.testing.assert_allclose(actual1, expected1)

    # R: eval_poly(poly1_unordered, c(1,1)) == 1.5
    actual_unordered = polynomial.eval_poly(POLY1_UNORDERED, NEWDATA1_SINGLE_OBS)
    assert isinstance(actual_unordered, float)
    np.testing.assert_allclose(actual_unordered, 1.5)

    # R: eval_poly(poly1, c(1,2)) == 1 - 1 + 2 + 0.5*1^2 == 2.5
    expected2 = 2.5
    actual2 = polynomial.eval_poly(POLY1, NEWDATA2_SINGLE_OBS)
    assert isinstance(actual2, float)
    np.testing.assert_allclose(actual2, expected2)

def test_multiple_poly_single_obs():
    # R: eval_poly(poly2, c(1,1))
    # PolyA: 1 - 1 + 1 = 1
    # PolyB: 2 + 3 - 2 = 3
    # R output: t(as.matrix(c(1,3))) -> array([[1,3]])
    # Python output: np.array([1,3]) (1D for single obs, multi poly)
    expected1 = np.array([1, 3])
    actual1 = polynomial.eval_poly(POLY2, NEWDATA1_SINGLE_OBS)
    assert actual1.ndim == 1
    np.testing.assert_allclose(actual1, expected1)

    # R: eval_poly(poly2_no_intercept, c(1,1))
    # PolyA: -1 + 1 = 0
    # PolyB:  3 - 2 = 1
    # R output: t(as.matrix(c(0,1))) -> array([[0,1]])
    # Python output: np.array([0,1])
    expected2 = np.array([0, 1])
    actual2 = polynomial.eval_poly(POLY2_NO_INTERCEPT, NEWDATA1_SINGLE_OBS)
    assert actual2.ndim == 1
    np.testing.assert_allclose(actual2, expected2)

    # R: eval_poly(poly2_unordered, c(1,1)) -> same as poly2, c(1,1) -> [1,3]
    expected3 = np.array([1,3])
    actual3 = polynomial.eval_poly(POLY2_UNORDERED, NEWDATA1_SINGLE_OBS)
    assert actual3.ndim == 1
    np.testing.assert_allclose(actual3, expected3)

def test_single_poly_multiple_obs():
    # R: eval_poly(poly1, rbind(c(1,2), c(1,1)))
    # Obs1 (1,2): 1 - 1 + 2 + 0.5*1^2 = 2.5
    # Obs2 (1,1): 1 - 1 + 1 + 0.5*1^2 = 1.5
    # R output: as.vector(c(2.5, 1.5)) -> array([2.5, 1.5])
    expected = np.array([2.5, 1.5])
    actual = polynomial.eval_poly(POLY1, NEWDATA_MULTI_OBS)
    assert actual.ndim == 1
    np.testing.assert_allclose(actual, expected)

def test_multiple_poly_multiple_obs():
    # R: eval_poly(poly2, rbind(c(1,2), c(1,1)))
    # Obs1 (1,2): PolyA: 1-1+2=2, PolyB: 2+3-4=1 -> R row: c(2,1)
    # Obs2 (1,1): PolyA: 1-1+1=1, PolyB: 2+3-2=3 -> R row: c(1,3)
    # R output: cbind(c(2,1), c(1,3)) -> array([[2,1],[1,3]])
    expected = np.array([[2,1], [1,3]])
    actual = polynomial.eval_poly(POLY2, NEWDATA_MULTI_OBS)
    assert actual.ndim == 2
    np.testing.assert_allclose(actual, expected)

    # R: eval_poly(poly2_no_intercept, rbind(c(1,2), c(1,1)))
    # Obs1 (1,2): PolyA: -1+2=1, PolyB: 3-4=-1 -> R row: c(1,-1)
    # Obs2 (1,1): PolyA: -1+1=0, PolyB: 3-2=1  -> R row: c(0,1)
    # R output: cbind(c(1,0), c(-1,1)) -> array([[1,-1],[0,1]])
    expected_ni = np.array([[1,-1], [0,1]])
    actual_ni = polynomial.eval_poly(POLY2_NO_INTERCEPT, NEWDATA_MULTI_OBS)
    assert actual_ni.ndim == 2
    np.testing.assert_allclose(actual_ni, expected_ni)

    # R: eval_poly(poly2_unordered, rbind(c(1,2), c(1,1))) -> same as poly2
    expected_uo = np.array([[2,1], [1,3]])
    actual_uo = polynomial.eval_poly(POLY2_UNORDERED, NEWDATA_MULTI_OBS)
    assert actual_uo.ndim == 2
    np.testing.assert_allclose(actual_uo, expected_uo)

def test_dataframe_input():
    # R: eval_poly(poly1, data.frame(X1=1, X2=1))
    # Python eval_poly expects numpy array, so convert DataFrame
    df_single_obs = pd.DataFrame({'X1': [1], 'X2': [1]})
    expected1 = 1.5
    actual1 = polynomial.eval_poly(POLY1, df_single_obs.to_numpy())
    assert isinstance(actual1, float)
    np.testing.assert_allclose(actual1, expected1)

    df_multi_obs = pd.DataFrame({'X1': [1,1], 'X2': [2,1]}) # Corresponds to NEWDATA_MULTI_OBS
    expected2 = np.array([2.5, 1.5])
    actual2 = polynomial.eval_poly(POLY1, df_multi_obs.to_numpy())
    assert actual2.ndim == 1
    np.testing.assert_allclose(actual2, expected2)

def test_high_order_first_no_intercept():
    # R: eval_poly(poly3, c(1,1))
    # poly3 labels = list(c(1,1,1),c(1),c(2)), values = c(1,2,3)
    # x1=1, x2=1: 1*(1^3) + 2*1 + 3*1 = 1 + 2 + 3 = 6
    expected = 6.0
    actual = polynomial.eval_poly(POLY3, NEWDATA1_SINGLE_OBS)
    assert isinstance(actual, float)
    np.testing.assert_allclose(actual, expected)

# === Tests for monomials=True ===
# Based on R test-monomials_true_various_scenarios.R

def test_monomials_true_single_poly_single_obs():
    # Poly1: (0,), (1,), (2,), (1,1) ; Values: 1, -1, 1, 0.5
    # Newdata: c(1,1) -> x1=1, x2=1
    # Term values:
    # (0,): 1 * 1 = 1
    # (1,): -1 * 1 = -1
    # (2,): 1 * 1 = 1
    # (1,1): 0.5 * 1^2 = 0.5
    # R output: as.vector(c(1,-1,1,0.5)) -> np.array([1,-1,1,0.5])
    expected = np.array([1, -1, 1, 0.5])
    actual = polynomial.eval_poly(POLY1, NEWDATA1_SINGLE_OBS, monomials=True)
    assert actual.ndim == 1 and actual.shape[0] == len(POLY1_LABELS)
    np.testing.assert_allclose(actual, expected)

def test_monomials_true_multiple_poly_single_obs():
    # Poly2: (0,), (1,), (2,); Values: [[1,2],[-1,3],[1,-2]]
    # Newdata: c(1,1) -> x1=1, x2=1
    # Term values expanded:
    # (0,): [1*1, 2*1] = [1,2]
    # (1,): [-1*1, 3*1] = [-1,3]
    # (2,): [1*1, -2*1] = [1,-2]
    # R output: matrix(c(1,-1,1, 2,3,-2), ncol=2, byrow=FALSE)
    # Python: (num_terms, num_polys) -> np.array([[1,2],[-1,3],[1,-2]])
    expected = np.array([[1,2], [-1,3], [1,-2]]) # Shape (3 terms, 2 polys)
    actual = polynomial.eval_poly(POLY2, NEWDATA1_SINGLE_OBS, monomials=True)
    assert actual.ndim == 2 and actual.shape == (len(POLY2_LABELS), POLY2_VALUES.shape[1])
    np.testing.assert_allclose(actual, expected)

def test_monomials_true_single_poly_multiple_obs():
    # Poly1: (0,), (1,), (2,), (1,1) ; Values: 1, -1, 1, 0.5
    # Newdata: rbind(c(1,2), c(1,1))
    # Obs1 (1,2): x1=1, x2=2. Terms: [1*1, -1*1, 1*2, 0.5*1^2] = [1, -1, 2, 0.5]
    # Obs2 (1,1): x1=1, x2=1. Terms: [1*1, -1*1, 1*1, 0.5*1^2] = [1, -1, 1, 0.5]
    # R output: rbind(c(1,-1,2,0.5), c(1,-1,1,0.5))
    # Python: (num_obs, num_terms) -> np.array([[1,-1,2,0.5],[1,-1,1,0.5]])
    expected = np.array([[1, -1, 2, 0.5], [1, -1, 1, 0.5]]) # Shape (2 obs, 4 terms)
    actual = polynomial.eval_poly(POLY1, NEWDATA_MULTI_OBS, monomials=True)
    assert actual.ndim == 2 and actual.shape == (NEWDATA_MULTI_OBS.shape[0], len(POLY1_LABELS))
    np.testing.assert_allclose(actual, expected)

def test_monomials_true_multiple_poly_multiple_obs():
    # Poly2, NEWDATA_MULTI_OBS
    # R output: array(dim=c(2,3,2)) (obs, terms, polys)
    # Obs1 (1,2): Terms for PolyA: [1,-1,2], Terms for PolyB: [2,3,-4]
    # Obs2 (1,1): Terms for PolyA: [1,-1,1], Terms for PolyB: [2,3,-2]
    # Python term_values_expanded: (n_obs, num_terms, num_polys)
    # term_values_expanded[0,:,0] = [1,-1,2] (Obs1, PolyA terms)
    # term_values_expanded[0,:,1] = [2,3,-4] (Obs1, PolyB terms)
    # term_values_expanded[1,:,0] = [1,-1,1] (Obs2, PolyA terms)
    # term_values_expanded[1,:,1] = [2,3,-2] (Obs2, PolyB terms)
    expected_array = np.zeros((2,3,2))
    expected_array[0,:,0] = [1,-1,2]; expected_array[0,:,1] = [2,3,-4]
    expected_array[1,:,0] = [1,-1,1]; expected_array[1,:,1] = [2,3,-2]
    actual = polynomial.eval_poly(POLY2, NEWDATA_MULTI_OBS, monomials=True)
    assert actual.ndim == 3 and actual.shape == (NEWDATA_MULTI_OBS.shape[0], len(POLY2_LABELS), POLY2_VALUES.shape[1])
    np.testing.assert_allclose(actual, expected_array)

# === Test: Sum of monomials=True output equals monomials=False output ===
# Based on R test-sum_monomials_equals_poly_eval.R

def check_sum_monomials_vs_poly_eval(poly, data):
    eval_true = polynomial.eval_poly(poly, data, monomials=True)
    eval_false = polynomial.eval_poly(poly, data, monomials=False) # Already squeezed

    # Sum eval_true over terms (axis depends on its shape due to squeezing)
    if eval_true.ndim == 1: # single_poly_single_obs -> (terms,)
        summed_eval_true = np.sum(eval_true, axis=0)
    elif eval_true.ndim == 2:
        # single_poly_multiple_obs -> (obs, terms) -> sum over axis 1
        # multiple_poly_single_obs -> (terms, polys) -> sum over axis 0
        if poly['values'].shape[1] == 1: # Single poly
            summed_eval_true = np.sum(eval_true, axis=1)
        else: # Multiple polys, single obs
            summed_eval_true = np.sum(eval_true, axis=0)
    elif eval_true.ndim == 3: # multiple_poly_multiple_obs -> (obs, terms, polys)
        summed_eval_true = np.sum(eval_true, axis=1)
    else:
        raise ValueError("Unexpected dimension for eval_true")
        
    np.testing.assert_allclose(summed_eval_true, eval_false,
                                err_msg=f"Sum of monomial values does not match direct polynomial evaluation. Poly labels: {poly['labels']}")

def test_monomials_sum_equals_poly_eval_all_scenarios():
    test_polys = {
        "P1": POLY1, "P1U": POLY1_UNORDERED,
        "P2": POLY2, "P2NI": POLY2_NO_INTERCEPT, "P2U": POLY2_UNORDERED,
        "P3": POLY3
    }
    test_datas = {
        "ND1_SO": NEWDATA1_SINGLE_OBS, "ND2_SO": NEWDATA2_SINGLE_OBS,
        "ND_MO": NEWDATA_MULTI_OBS
    }

    for pname, poly_dict in test_polys.items():
        for dname, data_arr in test_datas.items():
            # Check if poly is compatible with data (number of features)
            max_var_idx = 0
            for label in poly_dict['labels']:
                if label and label != (0,): # Not empty and not intercept
                    max_var_idx = max(max_var_idx, max(l for l in label if l!=0) ) # max var index in label
            
            if data_arr.shape[1] < max_var_idx : # newdata columns < max var index used
                # print(f"Skipping incompatible data {dname} for poly {pname}")
                continue 
            
            # print(f"Testing poly {pname} with data {dname}")
            check_sum_monomials_vs_poly_eval(poly_dict, data_arr)

```
