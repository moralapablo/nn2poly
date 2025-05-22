import pytest
import numpy as np
from nn2poly_py import model_parsers # The module to test

# === Mock Keras Classes ===

class MockKerasLayer:
    def __init__(self, input_dim, units, activation_name, name=None):
        self.input_dim = input_dim # Used to determine kernel shape
        self.units = units
        self.activation_name = activation_name
        self.name = name or f"mock_layer_{activation_name}"
        
        # For the first layer, input_shape is sometimes used by parsers to infer p_vars
        # For subsequent layers, the input_dim is inferred from previous layer's units.
        # The parser doesn't use self.input_shape directly, but needs get_weights()
        # which uses self.input_dim for the *kernel's* input dimension.
        self._input_shape_for_parser = (None, input_dim) # For first layer if parser needed it.

        # Actual weights for get_weights()
        # Kernel shape: (input_dim, units)
        # Bias shape: (units,)
        self._kernel = np.random.rand(self.input_dim, self.units)
        self._bias = np.random.rand(self.units)

        # Mocking the activation attribute as Keras layers might have it
        # The parser checks layer.activation.__name__ or layer.get_config()['activation']
        # Using get_config is more reliable for this mock.
        class MockActivation:
            def __init__(self, act_name):
                self.__name__ = act_name # For layer.activation.__name__
                self.name = act_name    # For layer.activation.name (Keras 3)
        
        self.activation = MockActivation(activation_name)


    def get_weights(self):
        # print(f"MockKerasLayer '{self.name}': get_weights() called. Kernel shape: {self._kernel.shape}, Bias shape: {self._bias.shape}")
        return [self._kernel, self._bias]

    def get_config(self):
        return {'activation': self.activation_name, 'units': self.units, 'name': self.name}

    @property
    def input_shape(self): # For first layer, if parser were to use it.
        return self._input_shape_for_parser


class MockKerasModel:
    def __init__(self, layers_list):
        self.layers = layers_list

# === Test for parse_keras_model ===

def test_parse_keras_model_skeletal():
    # Based on R keras_test_model():
    # Layer 1: input_dim=2, units=2, activation='tanh'
    # Layer 2: input_dim=2 (from L1 units), units=3, activation='relu' (maps to 'softplus')
    # Layer 3: input_dim=3 (from L2 units), units=2, activation='linear'
    
    mock_keras_layers = [
        MockKerasLayer(input_dim=2, units=2, activation_name='tanh', name='dense_1'),
        MockKerasLayer(input_dim=2, units=3, activation_name='relu', name='dense_2'), # input_dim is prev_layer.units
        MockKerasLayer(input_dim=3, units=2, activation_name='linear', name='dense_3') # input_dim is prev_layer.units
    ]
    mock_keras_model = MockKerasModel(mock_keras_layers)

    parsed_params = model_parsers.parse_keras_model(mock_keras_model)

    assert isinstance(parsed_params, dict)
    assert 'weights_list' in parsed_params
    assert isinstance(parsed_params['weights_list'], list)
    assert 'af_string_list' in parsed_params
    assert isinstance(parsed_params['af_string_list'], list)

    # Check activation functions (relu should map to softplus)
    assert parsed_params['af_string_list'] == ['tanh', 'softplus', 'linear']
    
    # Check number of weight matrices
    assert len(parsed_params['weights_list']) == 3

    # Check shapes of weight matrices (bias row + kernel rows)
    # Layer 1 weights: input_dim=2, units=2. Shape (1+2, 2) = (3,2)
    assert parsed_params['weights_list'][0].shape == (3, 2)
    # Layer 2 weights: input_dim=2 (from L1 units), units=3. Shape (1+2, 3) = (3,3)
    assert parsed_params['weights_list'][1].shape == (3, 3)
    # Layer 3 weights: input_dim=3 (from L2 units), units=2. Shape (1+3, 2) = (4,2)
    assert parsed_params['weights_list'][2].shape == (4, 2)


# === Mock PyTorch Classes ===

class MockTorchData: # To mock layer.weight.data and layer.bias.data
    def __init__(self, numpy_array):
        self._numpy_array = numpy_array
    def numpy(self):
        return self._numpy_array

class MockTorchParam: # To mock layer.weight and layer.bias
    def __init__(self, numpy_array):
        self.data = MockTorchData(numpy_array)

class MockTorchLayerLinear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features
        # PyTorch weight: (out_features, in_features)
        # PyTorch bias: (out_features,)
        self.weight = MockTorchParam(np.random.rand(self.out_features, self.in_features))
        self.bias = MockTorchParam(np.random.rand(self.out_features))
        self.__class__ = type('Linear', (object,), {'__name__': 'Linear'}) # Mocking __class__.__name__

class MockTorchActivation:
    def __init__(self, class_name_str): # e.g., 'Softplus', 'Tanh'
        self.class_name_str = class_name_str
        self.__class__ = type(class_name_str, (object,), {'__name__': class_name_str})

class MockTorchModel: # To provide the model.children() iterator
    def __init__(self, modules_list):
        self._modules_list = modules_list
    def children(self):
        return iter(self._modules_list)

# === Test for parse_pytorch_model ===

def test_parse_pytorch_model_skeletal():
    # Based on R luz_test_model():
    # Linear(2,2) -> Softplus
    # Linear(2,3) -> Softplus (R test used 'relu', mapped to 'softplus')
    # Linear(3,1) -> (Implicit Linear activation)
    
    mock_torch_modules = [
        MockTorchLayerLinear(in_features=2, out_features=2),
        MockTorchActivation('Softplus'),
        MockTorchLayerLinear(in_features=2, out_features=3), # in_features from previous Linear's out_features
        MockTorchActivation('ReLU'), # Will be mapped to 'softplus'
        MockTorchLayerLinear(in_features=3, out_features=1)  # in_features from previous Linear's out_features
        # Implicit linear activation for the last one
    ]
    mock_torch_model = MockTorchModel(mock_torch_modules)

    parsed_params = model_parsers.parse_pytorch_model(mock_torch_model)

    assert isinstance(parsed_params, dict)
    assert 'weights_list' in parsed_params
    assert isinstance(parsed_params['weights_list'], list)
    assert 'af_string_list' in parsed_params
    assert isinstance(parsed_params['af_string_list'], list)

    # Check activation functions (ReLU should map to softplus, last is linear)
    assert parsed_params['af_string_list'] == ['softplus', 'softplus', 'linear']
    
    # Check number of weight matrices (should correspond to Linear layers)
    assert len(parsed_params['weights_list']) == 3

    # Check shapes of weight matrices (bias row + kernel rows)
    # Kernel is transposed by parser from PyTorch's (out, in) to (in, out) before vstack.
    # L1 (Linear(2,2)): input_dim=2, units=2. Shape (1+2, 2) = (3,2)
    assert parsed_params['weights_list'][0].shape == (3, 2)
    # L2 (Linear(2,3)): input_dim=2, units=3. Shape (1+2, 3) = (3,3)
    assert parsed_params['weights_list'][1].shape == (3, 3)
    # L3 (Linear(3,1)): input_dim=3, units=1. Shape (1+3, 1) = (4,1)
    assert parsed_params['weights_list'][2].shape == (4, 1)

```
