# nn2poly-py: Transforming Neural Networks into Polynomials (Python Port)

<!-- Python badges placeholder - e.g., for GitHub Actions, PyPI version, etc. -->
<!-- [![Build Status](https://github.com/your_username/nn2poly_py/actions/workflows/python-package.yml/badge.svg)](https://github.com/your_username/nn2poly_py/actions/workflows/python-package.yml) -->
<!-- [![PyPI version](https://badge.fury.io/py/nn2poly-py.svg)](https://badge.fury.io/py/nn2poly-py) -->
[![DOI](https://img.shields.io/badge/doi-10.1016/j.neunet.2021.04.036-informational.svg)](https://doi.org/10.1016/j.neunet.2021.04.036)
[![DOI](https://img.shields.io/badge/doi-10.1109/TNNLS.2023.3330328-informational.svg)](https://doi.org/10.1109/TNNLS.2023.3330328)

The **nn2poly-py** package is a Python port of the R `nn2poly` method. It allows the transformation of trained deep feed-forward fully connected neural networks into a polynomial representation that predicts as similarly as possible to the original neural network. The obtained polynomial coefficients can be used to explain features (and their interactions) importance in the neural network, therefore working as a tool for interpretability or eXplainable Artificial Intelligence (XAI).

## Related Papers:

- Pablo Morala, J. Alexandra Cifuentes, Rosa E. Lillo, Iñaki Ucar (2021).
  "Towards a mathematical framework to inform neural network modelling via polynomial regression."
  _Neural Networks_, *142*, 57-72.
  doi: [10.1016/j.neunet.2021.04.036](https://doi.org/10.1016/j.neunet.2021.04.036)

- Pablo Morala, J. Alexandra Cifuentes, Rosa E. Lillo, Iñaki Ucar (2023).
  "NN2Poly: A Polynomial Representation for Deep Feed-Forward Artificial Neural Networks."
  _IEEE Transactions on Neural Networks and Learning Systems_, (Early Access).
  doi: [10.1109/TNNLS.2023.3330328](https://doi.org/10.1109/TNNLS.2023.3330328)

## Installation

It is highly recommended to use a virtual environment for installing Python packages.

### From Source (Editable Mode)

To install the package from a local clone of the repository in editable mode (allowing you to make changes to the source code that are immediately reflected in your environment):

```bash
git clone https://github.com/your_username/nn2poly_py.git  # Replace with actual URL
cd nn2poly_py
pip install -e .
```

### From PyPI (Placeholder)

Once the package is published to the Python Package Index (PyPI), you will be able to install it using:

```bash
pip install nn2poly-py  # Or the actual package name on PyPI
```

### Dependencies

The core `nn2poly-py` package requires the following Python libraries:
- NumPy (>=1.20)
- SymPy (>=1.8)
- SciPy (>=1.7)

These will be automatically installed if you use `pip`.

### Optional Dependencies for Model Parsers

To use the model parsing utilities for Keras and PyTorch, you need to install additional dependencies. You can install them using:

```bash
pip install "nn2poly-py[models]"
```
This will install `tensorflow` (for Keras) and `torch` (for PyTorch).

**Note:** The current model parsers (`parse_keras_model` and `parse_pytorch_model`) are skeletal implementations. They rely on the standard APIs of Keras and PyTorch but have not been extensively tested with diverse model architectures or library versions due to environment constraints during their initial development. They are best suited for simple, sequential models.

## Basic Python Usage Example

Here's a simple example of how to use `nn2poly-py` with manually defined weights and activation functions:

```python
import numpy as np
from nn2poly_py import nn2poly
from nn2poly_py.polynomial import predict_nn2poly # For prediction

# 1. Define the Neural Network Structure
# Example: 2-input -> Layer 1 (2 neurons, softplus) -> Layer 2 (1 neuron, linear)

# Weights for Layer 1 (2 inputs + bias, 2 neurons)
# Biases: [0.1, 0.2]
# Kernels: [[1.0, -1.0], [0.5, 0.5]] (input1_to_n1, input2_to_n1), (input1_to_n2, input2_to_n2)
# Combined: [[bias_n1, bias_n2], [w_i1n1, w_i1n2], [w_i2n1, w_i2n2]]
weights_l1 = np.array([
    [0.1, 0.2],    # Biases for neuron 1, neuron 2
    [1.0, -1.0],   # Weights from input 1 to neuron 1, neuron 2
    [0.5, 0.5]     # Weights from input 2 to neuron 1, neuron 2
])

# Weights for Layer 2 (2 inputs from L1 + bias, 1 neuron)
# Biases: [0.3]
# Kernels: [[1.0], [0.5]] (L1_n1_to_output, L1_n2_to_output)
weights_l2 = np.array([
    [0.3],         # Bias for output neuron
    [1.0],         # Weight from L1 neuron 1 to output neuron
    [0.5]          # Weight from L1 neuron 2 to output neuron
])

weights_list = [weights_l1, weights_l2]
af_string_list = ["softplus", "linear"] # Activations for Layer 1, Layer 2

# 2. Set Parameters for nn2poly
# Taylor orders for each layer's activation function
# 'softplus' order 2, 'linear' order 1 (linear is always order 1 internally)
taylor_orders = np.array([2, 1]) 
max_final_order = 2 # Desired maximum order of the final polynomial

# 3. Convert the NN to a Polynomial
# No precomputed_partitions needed for this small example if q_max_final_order is small.
# nn2poly will generate them if not provided.
# p_vars = 2 (from weights_l1.shape[0]-1)
# q_max_eff = min(2*1, 2) = 2
poly_representation = nn2poly(
    weights_list=weights_list,
    af_string_list=af_string_list,
    max_order=max_final_order,
    taylor_orders_param=taylor_orders, # Provide specific orders per layer
    keep_layers=False # Get only the final polynomial
)

# 4. Inspect the Polynomial Representation
# 'labels' are tuples: () for intercept, (1,) for x1, (1,1) for x1^2, etc.
# 'values' are coefficients (terms x neurons)
print("Polynomial Labels:", poly_representation['labels'])
print("Polynomial Coefficients:\n", poly_representation['values'])

# Example: For p_vars=2, max_order=2, labels might be:
# [(), (1,), (2,), (1,1), (1,2), (2,2)]
# Values shape would be (6, 1) for this network (1 output neuron)

# 5. Make Predictions using the Polynomial
# newdata should be (num_observations, num_features)
# For this example, num_features = 2 (x1, x2)
newdata_np = np.array([
    [1.0, 1.0],  # Observation 1
    [2.0, -1.0]  # Observation 2
])

predictions = predict_nn2poly(poly_representation, newdata_np)
print("Predictions:", predictions)

# If you used keep_layers=True:
# poly_representation_full = nn2poly(..., keep_layers=True)
# predictions_all_layers = predict_nn2poly(poly_representation_full, newdata_np, layers=None)
# print("Predictions for Layer 1 output:", predictions_all_layers['layer_1']['output'])
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
(Assuming the LICENSE file from the original R package is or will be included and is MIT. If not, this should be updated.)

## Contributing (Placeholder)

Contributions are welcome! Please read the contributing guidelines (link to be added) and submit pull requests to the `develop` branch (or as specified).

## Acknowledgements

This work is a Python port of the original R `nn2poly` package developed by S. L. N. T. de S. Jayalath, M. J. Reale, and R. P. Brent, inspired by their research papers.
