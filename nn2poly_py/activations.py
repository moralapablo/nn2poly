import numpy as np
import sympy
from sympy import Symbol, factorial, E, log as sympy_log

# Define common activation functions using NumPy for numerical evaluation
def linear(x):
    """Linear activation function."""
    return x

def sigmoid(x):
    """Sigmoid activation function."""
    if isinstance(x, sympy.Expr): # For symbolic calculation in Sympy
        return 1 / (1 + E**(-x))
    return 1 / (1 + np.exp(-x))

def tanh(x):
    """Hyperbolic tangent activation function."""
    if isinstance(x, sympy.Expr): # For symbolic calculation in Sympy
        return sympy.tanh(x)
    return np.tanh(x)

def softplus(x):
    """Softplus activation function."""
    if isinstance(x, sympy.Expr): # For symbolic calculation in Sympy
        return sympy_log(1 + E**x)
    return np.log(1 + np.exp(x))

# Map string names to function objects
_ACTIVATION_FUNCTIONS_MAP = {
    'linear': linear,
    'sigmoid': sigmoid,
    'tanh': tanh,
    'softplus': softplus,
}

def get_taylor_coefficients(func_name_str, order, around=0):
    """
    Calculates Taylor series coefficients for a given function around a point.

    Args:
        func_name_str (str): Name of the function (e.g., 'tanh', 'sigmoid').
        order (int): The maximum order of the Taylor expansion.
        around (float, optional): The point around which to expand. Defaults to 0.

    Returns:
        numpy.ndarray: An array of Taylor coefficients [c0, c1, ..., c_order],
                       where the series is sum(ck * (x-around)^k / k!).
                       The stored values are [f(a), f'(a), f''(a)/2!, ..., f^(n)(a)/n!].
    """
    if not isinstance(order, int) or order < 0:
        raise ValueError("Order must be a non-negative integer.")

    if func_name_str not in _ACTIVATION_FUNCTIONS_MAP:
        raise ValueError(f"Unknown activation function: {func_name_str}")

    x = Symbol('x')
    func_sympy = _ACTIVATION_FUNCTIONS_MAP[func_name_str](x)

    coeffs = []
    if func_name_str == 'linear':
        # f(x) = x. Around 'a': f(x) = a + 1*(x-a) + 0 + ...
        # Coeffs c_k = f^(k)(a)/k! are: [a, 1, 0, ..., 0]
        val_at_around = float(func_sympy.subs(x, around).evalf()) if isinstance(func_sympy, sympy.Expr) else func_sympy(around) # Should be 'around'
        coeffs.append(val_at_around)
        if order >= 1:
            coeffs.append(1.0)
        for _ in range(2, order + 1):
            coeffs.append(0.0)
    else:
        current_deriv = func_sympy
        for k in range(order + 1):
            val_at_around = current_deriv.subs(x, around).evalf()
            coeffs.append(float(val_at_around / factorial(k))) # c_k = f^(k)(a) / k!
            if k < order: # Avoid differentiating beyond the needed order
                current_deriv = sympy.diff(current_deriv, x)
    
    # Ensure the list has order + 1 elements.
    # This padding should not be necessary if the loop runs correctly.
    # However, for safety, if the loop structure was different:
    # while len(coeffs) < order + 1:
    #    coeffs.append(0.0)
    
    return np.array(coeffs[:order + 1])

if __name__ == '__main__':
    # Test cases
    print("Tanh (order 5, around 0):", get_taylor_coefficients('tanh', 5, 0))
    # Expected: [0, 1, 0, -1/3, 0, 2/15] approx [0.0, 1.0, 0.0, -0.33333333, 0.0, 0.13333333]
    # sympy.series(sympy.tanh(x), x, 0, 6).as_ordered_terms() gives: x - x**3/3 + 2*x**5/15 + O(x**6)
    # Coefficients f^(k)(0)/k! are:
    # k=0: tanh(0) = 0
    # k=1: sech(0)^2 = 1
    # k=2: -2*tanh(0)*sech(0)^2 = 0
    # k=3: (diff) / 3! = -1/3
    # k=4: (diff) / 4! = 0
    # k=5: (diff) / 5! = 2/15

    print("Sigmoid (order 5, around 0):", get_taylor_coefficients('sigmoid', 5, 0))
    # Expected: [1/2, 1/4, 0, -1/48, 0, 1/480] approx [0.5, 0.25, 0.0, -0.02083333, 0.0, 0.00208333]
    # sympy.series(1/(1+E**-x), x, 0, 6).as_ordered_terms() gives: 1/2 + x/4 - x**3/48 + x**5/480 + O(x**6)

    print("Softplus (order 4, around 0):", get_taylor_coefficients('softplus', 4, 0))
    # Expected: [log(2), 1/2, 1/8, 0, -1/192] approx [0.69314718, 0.5, 0.125, 0.0, -0.00520833]
    # sympy.series(sympy.log(1+E**x), x, 0, 5).as_ordered_terms() gives: log(2) + x/2 + x**2/8 - x**4/192 + O(x**5)
    
    print("Linear (order 3, around 0):", get_taylor_coefficients('linear', 3, 0))
    # Expected: [0, 1, 0, 0]
    
    print("Linear (order 3, around 2):", get_taylor_coefficients('linear', 3, 2))
    # Expected: [2, 1, 0, 0]

    # Test order 0
    print("Tanh (order 0, around 0):", get_taylor_coefficients('tanh', 0, 0)) # Expected: [0.0]
    print("Sigmoid (order 0, around 0):", get_taylor_coefficients('sigmoid', 0, 0)) # Expected: [0.5]

    # Test func_sympy(around) for linear
    x_sym = Symbol('x')
    lin_sym = _ACTIVATION_FUNCTIONS_MAP['linear'](x_sym) # this is just x_sym
    print("Linear Sympy type:", type(lin_sym))
    # For linear, func_sympy is just x. So subs(x, around) works.
    # val_at_around = float(lin_sym.subs(x, 0).evalf()) -> 0.0
    # val_at_around = float(lin_sym.subs(x, 2).evalf()) -> 2.0
    # This logic seems correct for linear in get_taylor_coefficients.
    
    # Check if sympy can be removed for linear
    # If func_name_str == 'linear':
    #   coeffs.append(float(around)) # c0 = f(a) = a
    #   if order >= 1: coeffs.append(1.0) # c1 = f'(a) = 1
    #   coeffs.extend([0.0] * (order - 1)) # c2, c3 ... are 0
    # This is what I implemented.

    # Test for the sympy.Expr check in activation functions
    print("Symbolic sigmoid(x):", sigmoid(x))
    print("Symbolic tanh(x):", tanh(x))
    print("Symbolic softplus(x):", softplus(x))
