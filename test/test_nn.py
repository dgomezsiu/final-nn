# TODO: import dependencies and write unit tests below


import numpy as np
import pytest

from nn import nn
from nn import preprocess

nn_arch = [
    {'input_dim': 2, 'output_dim': 4, 'activation': 'relu'},
    {'input_dim': 4, 'output_dim': 2, 'activation': 'sigmoid'}
]
lr = 0.01
seed = 42
batch_size = 32
epochs = 10
loss_function = "binary_cross_entropy"

nn = nn.NeuralNetwork(nn_arch, lr, seed, batch_size, epochs, loss_function)

def test_single_forward():
    W_curr = np.array([[1, -1], [0, 1]])
    b_curr = np.array([[0], [0]])
    A_prev = np.array([[2], [-3]])
    activation = 'relu'
    A_curr, Z_curr = nn._single_forward(W_curr, b_curr, A_prev, activation)

    expected_Z_curr = np.dot(W_curr, A_prev) + b_curr
    expected_A_curr = np.maximum(0, expected_Z_curr)
    
    np.testing.assert_array_equal(Z_curr, expected_Z_curr)
    np.testing.assert_array_equal(A_curr, expected_A_curr)

def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    pass