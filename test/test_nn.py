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
    X = np.random.randn(4, 2)

    # Execute forward pass
    output, cache = nn.forward(X.T)

    # Validate output shape
    assert output.shape == (4, 2)

def test_single_backprop():
    W_curr = np.ones((2, 4))   
    b_curr = np.zeros((2, 1))
    Z_curr = np.random.randn(2, 3)
    A_prev = np.random.randn(4, 3)
    dA_curr = np.random.randn(2, 3)
    activation_curr = 'relu'

    dA_prev, dW_curr, db_curr = nn._single_backprop(W_curr, Z_curr, A_prev, dA_curr, activation_curr)

    # Assertions to verify the gradients' shapes match expected dimensions
    assert dA_prev.shape == A_prev.shape
    assert dW_curr.shape == W_curr.shape
    assert db_curr.shape == b_curr.shape

def test_predict():
    X = np.random.randn(4,2)

    y_hat = nn.predict(X)
    expected_shape = (4,2)
    assert y_hat.shape == expected_shape

def test_binary_cross_entropy():
    y_hat = np.array([0.9, 0.2, 0.1, 0.4])
    y = np.array([1, 0, 0, 1])

    # Expected loss
    expected_loss = -np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    
    loss = nn._binary_cross_entropy(y, y_hat)
    assert np.isclose(loss, expected_loss)

def test_binary_cross_entropy_backprop():
    y_hat = np.array([0.9, 0.2, 0.1, 0.4]).reshape(1, -1)
    y = np.array([1, 0, 0, 1]).reshape(1, -1)

    # expected gradient
    expected_gradient = - (np.divide(y, y_hat) - np.divide(1 - y, 1 - y_hat))
    
    gradient = nn._binary_cross_entropy_backprop(y, y_hat)
    assert len(gradient) == 4

def test_mean_squared_error():
    y_hat = np.array([1.5, 2.5, 3.5])
    y = np.array([1, 2, 3])

    # expected loss
    expected_loss = np.mean(0.5 * (y_hat - y) ** 2)
    
    loss = nn._mean_squared_error(y, y_hat)
    assert np.isclose(loss, expected_loss)

def test_mean_squared_error_backprop():
    y_hat = np.array([1.5, 2.5, 3.5]).reshape(1, -1)
    y = np.array([1, 2, 3]).reshape(1, -1)

    # expected gradient
    expected_gradient = (1 / len(y)) * (y_hat - y)
    
    gradient = nn._mean_squared_error_backprop(y, y_hat)
    assert len(gradient) == 3

def test_sample_seqs():
    seqs = ['A', 'B', 'C', 'D']
    labels = [True, True, False, False]
    
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)
    
    # check balance
    assert sum(sampled_labels) == len(sampled_labels) / 2

    labels = [True, True, False, True]  # Only one False
    
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)
    
    # expect one duplicate in the negative samples
    assert len(set(sampled_seqs)) > len(sampled_seqs) / 2
    assert sum(sampled_labels) == len(sampled_labels) / 2

def test_one_hot_encode_seqs():
    seq_arr = ['AGA']
    expected_output = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]])

    assert np.array_equal(preprocess.one_hot_encode_seqs(seq_arr), expected_output)


    seq_arr = ['A', 'GT', 'ACG']
    expected_output = [[1, 0, 0, 0], [0, 0, 0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]]
    print(preprocess.one_hot_encode_seqs(seq_arr))
    assert np.all(preprocess.one_hot_encode_seqs(seq_arr), expected_output)