import numpy as np

def compute_mse(y, tx, w):
    """Returns Mean Squared Error of a prediction"""
    e = y - tx @ w
    return e @ e / (2*len(y))

def compute_gradient(y, tx, w):
    """Returns gradient of loss function"""
    e = y - tx @ w
    return - tx.T @ e / len(y)

def sigmoid(t):
    """Applies sigmoid function to a variable"""
    return 1 / (1 + np.exp(-t))

