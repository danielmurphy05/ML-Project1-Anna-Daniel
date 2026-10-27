import numpy as np


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def compute_mse(y, tx, w):
    """Compute the MSE loss (with the 1/2 factor) of weights w.

    Args:
        y: targets, shape (N,)
        tx: features with bias column, shape (N, D)
        w: weights, shape (D,)

    Returns:
        MSE loss (scalar)
    """
    e = y - tx @ w
    return e @ e / (2 * len(y))


def compute_gradient(y, tx, w):
    """Compute the gradient of the MSE loss at w.

    Args:
        y: targets, shape (N,)
        tx: features with bias column, shape (N, D)
        w: weights, shape (D,)

    Returns:
        gradient, shape (D,)
    """
    e = y - tx @ w
    return -tx.T @ e / len(y)


def sigmoid(t):
    """Apply the sigmoid function element-wise."""
    return 1 / (1 + np.exp(-t))


def compute_logistic_loss(y, tx, w):
    """Logistic loss (mean negative log-likelihood) at w, for y in {0, 1}."""
    t = tx @ w
    return np.mean(np.logaddexp(0, t) - y * t)


def compute_logistic_gradient(y, tx, w):
    """Gradient of the logistic loss (mean negative log-likelihood) at w."""
    return tx.T @ (sigmoid(tx @ w) - y) / len(y)


# ---------------------------------------------------------------------------
# Required functions
# ---------------------------------------------------------------------------


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.

    Args:
        y: targets, shape (N,)
        tx: features with bias column, shape (N, D)
        initial_w: starting weights, shape (D,)
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        w: final weights, shape (D,)
        loss: MSE loss at the final w (scalar)
    """
    w = initial_w
    for _ in range(max_iters):
        g = compute_gradient(y, tx, w)
        w = w - gamma * g
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent (batch size 1).

    At each iteration, one sample is drawn uniformly at random and the
    weights are updated using the gradient of the MSE on that sample only.

    Args:
        y: targets, shape (N,)
        tx: features with bias column, shape (N, D)
        initial_w: starting weights, shape (D,)
        max_iters: number of SGD steps (one sample per step)
        gamma: step size

    Returns:
        w: final weights, shape (D,)
        loss: MSE loss over the full dataset at the final w (scalar)
    """
    w = initial_w
    for _ in range(max_iters):
        n = np.random.randint(len(y))
        g = compute_gradient(y[n : n + 1], tx[n : n + 1], w)
        w = w - gamma * g
    loss = compute_mse(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Solves (tx^T tx) w = tx^T y for the weights that minimise the MSE.

    Args:
        y: targets, shape (N,)
        tx: features with bias column, shape (N, D)

    Returns:
        w: optimal weights, shape (D,)
        loss: MSE loss at w (scalar)
    """
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the normal equations.

    Solves (tx^T tx + 2 N lambda_ I) w = tx^T y, which minimises
    MSE(w) + lambda_ * ||w||^2.

    Args:
        y: targets, shape (N,)
        tx: features with bias column, shape (N, D)
        lambda_: regularisation strength (>= 0)

    Returns:
        w: optimal weights, shape (D,)
        loss: MSE loss at w, without the penalty term (scalar)
    """
    c = 2 * len(y) * lambda_
    A = tx.T @ tx + c * np.eye(tx.shape[1])
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    loss = compute_mse(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent (y in {0, 1}).

    Args:
        y: binary labels in {0, 1}, shape (N,)
        tx: features with bias column, shape (N, D)
        initial_w: starting weights, shape (D,)
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        w: final weights, shape (D,)
        loss: logistic loss (mean negative log-likelihood) at the final w (scalar)
    """
    w = initial_w
    for _ in range(max_iters):
        g = compute_logistic_gradient(y, tx, w)
        w = w - gamma * g
    loss = compute_logistic_loss(y, tx, w)
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularised logistic regression using gradient descent (y in {0, 1}).

    Minimises logistic loss + lambda_ * ||w||^2.

    Args:
        y: binary labels in {0, 1}, shape (N,)
        tx: features with bias column, shape (N, D)
        lambda_: regularisation strength (>= 0)
        initial_w: starting weights, shape (D,)
        max_iters: number of gradient steps
        gamma: step size

    Returns:
        w: final weights, shape (D,)
        loss: logistic loss at the final w, without the penalty term (scalar)
    """
    w = initial_w
    for _ in range(max_iters):
        g = compute_logistic_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * g
    loss = compute_logistic_loss(y, tx, w)
    return w, loss
