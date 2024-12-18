import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Generate synthetic data
X_train, y_train = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)


# Add polynomial features (x1^2, x2^2, x1 * x2)
def add_polynomial_features(X):
    x1 = X[:, 0]
    x2 = X[:, 1]
    X_poly = np.column_stack((x1, x2, x1 ** 2, x2 ** 2, x1 * x2))
    return X_poly


# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# Compute cost with regularization
def compute_cost_with_regularization(X, y, w, b, lambda_=1):
    m = X.shape[0]
    f_wb = sigmoid(np.dot(X, w) + b)
    cost = (-1 / m) * (np.dot(y, np.log(f_wb)) + np.dot(1 - y, np.log(1 - f_wb)))
    reg_cost = (lambda_ / (2 * m)) * np.sum(w ** 2)
    return cost + reg_cost


# Compute gradients with regularization
def compute_gradient(X, y, w, b, lambda_=1):
    m = X.shape[0]
    z = np.dot(X, w) + b
    f_wb = sigmoid(z)
    error = f_wb - y
    dj_dw = (1 / m) * np.dot(X.T, error) + (lambda_ / m) * w  # Regularize w (not b)
    dj_db = (1 / m) * np.sum(error)
    return dj_dw, dj_db


# Function to capture decision boundary at different iterations
def store_decision_boundary(X, y, w, b, snapshots, iteration):
    """
    Store the state of the decision boundary at the current iteration.
    Args:
        X (ndarray): Input data, used to plot the decision boundary.
        y (ndarray): Labels of the data points.
        w (ndarray): Weights of the model.
        b (scalar): Bias of the model.
        snapshots (list): List to store decision boundary snapshots.
        iteration (int): Current iteration number.
    """
    snapshots.append((w.copy(), b, iteration))


# Function to plot all stored decision boundaries in subplots
def plot_all_decision_boundaries(X, y, snapshots):
    """
    Plot all decision boundaries stored in the snapshots list in a subplot grid.
    Args:
        X (ndarray): Input data, used to plot the decision boundary.
        y (ndarray): Labels of the data points.
        snapshots (list): List of (w, b, iteration) tuples.
    """
    num_plots = len(snapshots)
    cols = 3  # Number of columns in the subplot grid
    rows = (num_plots // cols) + (num_plots % cols > 0)  # Compute number of rows

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.ravel()  # Flatten axes array for easy iteration

    for i, (w, b, iteration) in enumerate(snapshots):
        ax = axes[i]
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))

        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_points_poly = add_polynomial_features(grid_points)
        Z = sigmoid(np.dot(grid_points_poly, w) + b)
        Z = Z.reshape(xx.shape)

        ax.contourf(xx, yy, Z, alpha=0.5, levels=np.linspace(0, 1, 100), cmap=plt.cm.RdYlBu)
        ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu, s=50)

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_title(f"Iteration {iteration}")

    # Remove any empty subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# Gradient descent with boundary snapshots
def gradient_descent_with_snapshots(X, y, w, b, alpha, num_iters, lambda_):
    J_history = []
    snapshots = []  # To store decision boundary snapshots

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b, lambda_)

        # Update the parameters
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Store the cost for each iteration (optional, to monitor the cost)
        if i % 100 == 0:  # Log every 100 iterations
            cost = compute_cost_with_regularization(X, y, w, b, lambda_)
            J_history.append(cost)
            print(f"Iteration {i}: Cost = {cost}")

        # Store decision boundary snapshots every 500 iterations
        if i % 500 == 0:
            store_decision_boundary(X, y, w, b, snapshots, i)

    # Capture the final decision boundary
    store_decision_boundary(X, y, w, b, snapshots, num_iters)

    return w, b, J_history, snapshots


# Initialize parameters
X_train_poly = add_polynomial_features(X_train)
w = np.zeros(X_train_poly.shape[1])
b = 0
alpha = 0.01
num_iters = 5000
lambda_ = 0.1

# Run gradient descent with snapshots
w, b, J_history, snapshots = gradient_descent_with_snapshots(X_train_poly, y_train, w, b, alpha, num_iters, lambda_)

# Plot all decision boundaries after training
plot_all_decision_boundaries(X_train, y_train, snapshots)
