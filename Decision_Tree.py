###############################################
# Generalized Decision Tree and Random Forest
###############################################

# Libraries used in this implementation
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from graphviz import Digraph
from sklearn.datasets import make_regression


# --------------------------------------------
# Node Class
# --------------------------------------------
class Node:
    """
    Represents a single node in the Decision Tree.

    Attributes:
        threshold (float): The splitting value for the feature.
        feature (int): Index of the feature used for splitting.
        left (Node): Left child node.
        right (Node): Right child node.
        prediction (float): Prediction value for leaf nodes.
    """

    def __init__(self, threshold=None, feature=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


# --------------------------------------------
# Helper Functions
# --------------------------------------------
def variance(X, node_indices, feature, simple=1):
    """
    Calculate variance for a given feature within the node indices.

    Args:
        X (ndarray): Feature matrix.
        node_indices (ndarray): Indices of the current node's data.
        feature (int): Feature index.
        simple (int): 1 for sample variance, 0 for population variance.

    Returns:
        float: Calculated variance.
    """
    values = X[node_indices][feature]
    mean = np.mean(values)
    n = len(node_indices)
    var = sum((values - mean) ** 2)
    return var / (n - 1) if simple and n > 1 else var / n


def find_best_threshold(X, y, node_indices, feature):
    """
    Identify the best threshold for a feature to split the data.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Target values.
        node_indices (ndarray): Indices of the current node's data.
        feature (int): Feature index to evaluate.

    Returns:
        tuple: Best threshold and associated variance reduction.
    """
    values = X[node_indices, feature]
    sorted_indices = node_indices[np.argsort(values)]
    sorted_values = X[sorted_indices, feature]
    sorted_y = y[sorted_indices]

    left_y = deque()
    right_y = deque(sorted_y)

    n = len(node_indices)
    root_var = np.var(sorted_y)
    best_threshold = None
    best_reduction = -1

    for i in range(1, n):
        left_y.append(right_y.popleft())
        if sorted_values[i] == sorted_values[i - 1]:
            continue

        var_left = np.var(left_y) if len(left_y) > 1 else 0
        var_right = np.var(right_y) if len(right_y) > 1 else 0

        W_left = len(left_y) / n
        W_right = len(right_y) / n
        variance_reduction = root_var - (W_left * var_left + W_right * var_right)

        if variance_reduction > best_reduction:
            best_threshold = 0.5 * (sorted_values[i] + sorted_values[i - 1])
            best_reduction = variance_reduction

    return best_threshold, best_reduction


def find_best_split(X, y, node_indices):
    """
    Find the best feature and threshold for splitting the data.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Target values.
        node_indices (ndarray): Indices of the current node's data.

    Returns:
        tuple: Best feature, threshold, and variance reduction.
    """
    n = X.shape[1]
    best_feature = None
    best_threshold = None
    best_reduction = -1

    for feature in range(n):
        threshold, reduction = find_best_threshold(X, y, node_indices, feature)
        if reduction > best_reduction:
            best_feature = feature
            best_threshold = threshold
            best_reduction = reduction

    return best_feature, best_threshold, best_reduction


def create_leaf_node(y_values):
    """
    Create a leaf node for the decision tree.

    Args:
        y_values (ndarray): Target values in the node.

    Returns:
        Node: Leaf node with the mean prediction.
    """
    mean = np.mean(y_values)
    return Node(prediction=mean)


# --------------------------------------------
# Decision Tree Functions
# --------------------------------------------
def build_tree(X, y, node_indices=None, depth=0, max_depth=10, min_samples_split=2):
    """
    Build a Decision Tree recursively.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Target values.
        node_indices (ndarray): Indices of the current node's data.
        depth (int): Current depth of the tree.
        max_depth (int): Maximum depth of the tree.
        min_samples_split (int): Minimum samples to split a node.

    Returns:
        Node: Root node of the tree.
    """
    if node_indices is None:
        node_indices = np.arange(X.shape[0])

    if len(node_indices) < min_samples_split or depth >= max_depth or np.all(y[node_indices] == y[node_indices][0]):
        return create_leaf_node(y[node_indices])

    feature, threshold, reduction = find_best_split(X, y, node_indices)

    if feature is None or reduction <= 0.05:
        return create_leaf_node(y[node_indices])

    left_indices = node_indices[X[node_indices, feature] <= threshold]
    right_indices = node_indices[X[node_indices, feature] > threshold]

    left_child = build_tree(X, y, left_indices, depth + 1, max_depth, min_samples_split)
    right_child = build_tree(X, y, right_indices, depth + 1, max_depth, min_samples_split)

    return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)


def predict_single(x, root):
    """
    Predict a single sample using the Decision Tree.

    Args:
        x (ndarray): Feature vector of the sample.
        root (Node): Root node of the tree.

    Returns:
        float: Prediction for the sample.
    """
    node = root
    while node.feature is not None:
        node = node.left if x[node.feature] <= node.threshold else node.right
    return node.prediction


def predict(X, root):
    """
    Predict multiple samples using the Decision Tree.

    Args:
        X (ndarray): Feature matrix.
        root (Node): Root node of the tree.

    Returns:
        ndarray: Predictions for the samples.
    """
    return np.array([predict_single(sample, root) for sample in X])


# --------------------------------------------
# Visualization Functions
# --------------------------------------------
def print_tree_console(node, depth=0):
    """
    Print the Decision Tree structure to the console.

    Args:
        node (Node): Root node of the tree.
        depth (int): Current depth in the tree.
    """
    if node.feature is None:
        print(f"{'|   ' * depth}Predict: {node.prediction:.3f}")
        return

    print(f"{'|   ' * depth}Feature {node.feature} <= {node.threshold:.3f}")
    print_tree_console(node.left, depth + 1)
    print(f"{'|   ' * depth}Feature {node.feature} > {node.threshold:.3f}")
    print_tree_console(node.right, depth + 1)


def plot_tree_using_matplot(node, depth=0, pos=(0.5, 1), width=1, ax=None):
    """
    Plot the Decision Tree structure using Matplotlib.

    Args:
        node (Node): Root node of the tree.
        depth (int): Current depth in the tree.
        pos (tuple): Position of the current node.
        width (float): Width of the plot.
        ax (Axes): Matplotlib Axes instance.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")

    x, y = pos
    if node.feature is None:
        ax.text(x, y, f"Predict: {node.prediction:.3f}", ha="center",
                bbox=dict(boxstyle="round", facecolor="lightgray"))
    else:
        ax.text(x, y, f"Feature {node.feature} <= {node.threshold:.3f}", ha="center",
                bbox=dict(boxstyle="round", facecolor="lightblue"))

        dx = width / 2 ** (depth + 1)
        dy = 0.1

        left_pos = (x - dx, y - dy)
        ax.plot([x, left_pos[0]], [y, left_pos[1]], 'k-')
        ax.text((x + left_pos[0]) / 2, (y + left_pos[1]) / 2, "Yes", color="green", fontsize=10, ha="center",
                va="center")
        plot_tree_using_matplot(node.left, depth + 1, left_pos, width, ax)

        right_pos = (x + dx, y - dy)
        ax.plot([x, right_pos[0]], [y, right_pos[1]], 'k-')
        ax.text((x + right_pos[0]) / 2, (y + right_pos[1]) / 2, "No", color="red", fontsize=10, ha="center",
                va="center")
        plot_tree_using_matplot(node.right, depth + 1, right_pos, width, ax)


def export_tree_to_dot(node, dot=None, node_id=0, depth=0):
    """
    Export the Decision Tree structure to Graphviz DOT format.

    Args:
        node (Node): Root node of the tree.
        dot (Digraph): Graphviz Digraph instance.
        node_id (int): Current node ID.
        depth (int): Current depth in the tree.

    Returns:
        Digraph: Graphviz Digraph representing the tree.
    """
    if dot is None:
        dot = Digraph()

    current_id = node_id
    if node.feature is None:
        dot.node(str(current_id), f"Predict: {node.prediction:.3f}", shape="box")
    else:
        dot.node(str(current_id), f"Feature {node.feature} <= {node.threshold:.3f}")
        left_id = current_id + 1
        dot.edge(str(current_id), str(left_id), label="Yes")
        export_tree_to_dot(node.left, dot, left_id, depth + 1)

        right_id = left_id + (2 ** depth)
        dot.edge(str(current_id), str(right_id), label="No")
        export_tree_to_dot(node.right, dot, right_id, depth + 1)

    return dot


# --------------------------------------------
# Example Usage
# --------------------------------------------


# X_train, y_train = make_regression(n_samples=1000, n_features=10, n_informative=10)
#
# # Build and visualize a single tree
# root = build_tree(X_train, y_train, max_depth=5, min_samples_split=2)
# plot_tree_using_matplot(root)
# plt.show()
#
# # Predict with a single tree
# predictions = predict(X_train, root)
# print("Single Tree Predictions:", predictions)
