#-----------------------------------------------
# This is the Generalized version of the Decision Tree
#-----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from graphviz import Digraph
from sklearn.datasets import make_regression
# This class to store the Decision Tree:

class Node:
    def __init__(self, threshold=None, feature=None, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction


def variance(X, node_indices, feature, simple=1):
    values = X[node_indices][feature]
    mean = np.mean(values)
    n = len(node_indices)
    var = sum((values - mean) ** 2)
    # there are two types of variance:
    # simple and population, so here I can choose which one I want:
    if simple == 1:
        return var / (n - 1) if n > 1 else 0
    return var / n




def find_best_threshold(X, y, node_indices, feature):
    # Extract feature values and corresponding target values
    values = X[node_indices , feature]
    sorted_indices = node_indices[np.argsort(values)]  # Sort based on feature values
    sorted_values = X[sorted_indices , feature]
    sorted_y = y[sorted_indices]

    # Initialize left and right subsets
    left_y = deque()  # Deque to store `y` values for the left subset
    right_y = deque(sorted_y)  # Initially, all values are in the right subset

    n = len(node_indices)
    root_var = np.var(sorted_y)  # Variance of the entire node
    best_threshold = None
    best_reduction = -1

    # Iterate over possible thresholds
    for i in range(1, n):  # Iterate over sorted values
        # Move one element from right_y to left_y
        left_y.append(right_y.popleft())

        # Skip duplicate feature values to avoid redundant splits
        if sorted_values[i] == sorted_values[i - 1]:
            continue

        # Calculate variance for left and right subsets
        var_left = np.var(left_y) if len(left_y) > 1 else 0
        var_right = np.var(right_y) if len(right_y) > 1 else 0

        # Calculate weighted variance reduction
        W_left = len(left_y) / n
        W_right = len(right_y) / n
        variance_reduction = root_var - (W_left * var_left + W_right * var_right)

        # Update the best threshold if this split is better
        if variance_reduction > best_reduction:
            best_threshold = 0.5 * (sorted_values[i] + sorted_values[i - 1])
            best_reduction = variance_reduction

    return best_threshold, best_reduction

def find_best_split(X, y, node_indices):
    n = X.shape[1]
    best_feature = None
    best_reduction = -1
    best_threshold = None
    for feature in range(n):
        threshold, reduction = find_best_threshold(X, y, node_indices, feature)
        if reduction > best_reduction:
            best_reduction = reduction
            best_feature = feature
            best_threshold = threshold

    return best_feature, best_threshold, best_reduction


def create_leaf_node(y_values):
    mean = np.mean(y_values)
    return Node(prediction=mean)


def build_tree(X, y, node_indices=None, depth=0, max_depth=10, min_samples_split=2):
    if node_indices is None:
        node_indices = np.arange(X.shape[0])
    # print(f"Node Indices: {node_indices}, X shape is : {X.shape}")
    # Stopping criteria:
    # 1. Check if we have fewer samples that min_samples_split:
    if len(node_indices) < min_samples_split:
        return create_leaf_node(y[node_indices])
    # 2. Check if max_depth is reached:
    if max_depth is not None and depth >= max_depth:
        return create_leaf_node(y[node_indices])
    # 3. Check if there's no variance in y:
    if np.all(y[node_indices] == y[node_indices][0]):
        return create_leaf_node(y[node_indices])

    # Find the best split
    feature, threshold, reduction = find_best_split(X, y, node_indices)

    # If no good split is found (reduction is too small), make a leaf node
    if feature is None or reduction <= 0.05:
        return create_leaf_node(y[node_indices])

    # Partition the tree:
    left_indices = node_indices[X[node_indices, feature] <= threshold]
    right_indices = node_indices[X[node_indices, feature] > threshold]

    left_child = build_tree(X, y, left_indices, depth + 1, max_depth, min_samples_split)
    right_child = build_tree(X, y, right_indices, depth + 1, max_depth, min_samples_split)

    #Create internal node
    return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)


# Prediction
def predict_single(x, root):
    node = root
    while node.feature is not None:
        if x[node.feature] <= node.threshold:
            node = node.left
        else:
            node = node.right

    return node.prediction


def predict(X, root):
    predictions = []
    m = X.shape[0]
    for sample in X:
        predictions.append(predict_single(sample, root))
    return np.array(predictions)


# Random Forest
def build_random_forest(X, y, n_trees=10, max_depth=10, min_samples_split=2):
    trees = []
    n = X.shape[0]
    for _ in range(n_trees):
        random_indices = np.random.choice(n, n, replace=True)
        X_random = X[random_indices]
        y_random = y[random_indices]

        tree = build_tree(X_random, y_random, max_depth=max_depth, min_samples_split=min_samples_split)
        trees.append(tree)

    return trees


def random_forest_predict(X, trees):
    all_predictions = [predict(X, tree) for tree in trees]
    return np.mean(all_predictions, axis=0)


# Printing Functions :

def print_tree_console(node, depth=0):
    # If it's a leaf node
    if node.feature is None:
        print(f"{'|   ' * depth}Predict: {node.prediction:.3f}")
        return

    # Internal node
    print(f"{'|   ' * depth}Feature {node.feature} <= {node.threshold:.3f}")
    print_tree_console(node.left, depth + 1)
    print(f"{'|   ' * depth}Feature {node.feature} > {node.threshold:.3f}")
    print_tree_console(node.right, depth + 1)


def plot_tree_using_matplot(node, depth=0, pos=(0.5, 1), width=1, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")

    x, y = pos
    if node.feature is None:
        # Leaf node
        ax.text(x, y, f"Predict: {node.prediction:.3f}", ha="center",
                bbox=dict(boxstyle="round", facecolor="lightgray"))
    else:
        # Decision node
        ax.text(x, y, f"Feature {node.feature} <= {node.threshold:.3f}", ha="center",
                bbox=dict(boxstyle="round", facecolor="lightblue"))

        dx = width / 2 ** (depth + 1)  # Horizontal spacing
        dy = 0.1  # Vertical spacing

        # Plot left child (Yes edge)
        left_pos = (x - dx, y - dy)
        ax.plot([x, left_pos[0]], [y, left_pos[1]], 'k-')  # Draw edge
        ax.text((x + left_pos[0]) / 2, (y + left_pos[1]) / 2, "Yes", color="green",
                fontsize=10, ha="center", va="center")
        plot_tree_using_matplot(node.left, depth + 1, left_pos, width, ax)

        # Plot right child (No edge)
        right_pos = (x + dx, y - dy)
        ax.plot([x, right_pos[0]], [y, right_pos[1]], 'k-')  # Draw edge
        ax.text((x + right_pos[0]) / 2, (y + right_pos[1]) / 2, "No", color="red",
                fontsize=10, ha="center", va="center")
        plot_tree_using_matplot(node.right, depth + 1, right_pos, width, ax)

    if ax is not None:
        return ax


def export_tree_to_dot(node, dot=None, node_id=0, depth=0):
    if dot is None:
        dot = Digraph()

    current_id = node_id
    if node.feature is None:
        # Leaf node
        dot.node(str(current_id), f"Predict: {node.prediction:.3f}", shape="box")
    else:
        # Internal node
        dot.node(str(current_id), f"Feature {node.feature} <= {node.threshold:.3f}")
        left_id = current_id + 1
        dot.edge(str(current_id), str(left_id), label="Yes")
        export_tree_to_dot(node.left, dot, left_id, depth + 1)

        right_id = left_id + (2 ** depth)
        dot.edge(str(current_id), str(right_id), label="No")
        export_tree_to_dot(node.right, dot, right_id, depth + 1)

    return dot


X_train, y_train = make_regression(n_samples=1000, n_features=10, n_informative=10)

# Build a single tree
root = build_tree(X_train, y_train, max_depth=5, min_samples_split=2)

# Testing the print functionality:

# Console:
# print_tree_console(root, 0)

# Graphviz
dot = export_tree_to_dot(root)
dot.render("decision_tree", format="png", cleanup=True)

# Matplot:
plot_tree_using_matplot(root)
plt.show()


# Predict using the tree
predictions = predict(X_train, root)
print("Single Tree Predictions:", predictions)

# Build a random forest
forest = build_random_forest(X_train, y_train, n_trees=5, max_depth=3, min_samples_split=2)

# Predict using the random forest
rf_predictions = random_forest_predict(X_train, forest)
print("Random Forest Predictions:", rf_predictions)










