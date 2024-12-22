# --------------------------------------------
# Libraries
# --------------------------------------------
import numpy as np
from Decision_Tree import Node, build_tree, predict
from sklearn.datasets import make_regression


# --------------------------------------------
# Random Forest Functions
# --------------------------------------------
def build_random_forest(X, y, n_trees=10, max_depth=10, min_samples_split=2, max_features=None):
    """
    Build a Random Forest with random sampling of data and features.

    Args:
        X (ndarray): Feature matrix.
        y (ndarray): Target values.
        n_trees (int): Number of trees in the forest.
        max_depth (int): Maximum depth of each tree.
        min_samples_split (int): Minimum samples to split a node.
        max_features (int): Number of features to consider per split.

    Returns:
        list: List of trees and their selected feature subsets.
    """
    trees = []
    n_samples, n_features = X.shape

    if max_features is None:
        max_features = int(np.sqrt(n_features))

    for _ in range(n_trees):
        random_indices = np.random.choice(n_samples, n_samples, replace=True)
        X_random = X[random_indices]
        y_random = y[random_indices]

        feature_indices = np.random.choice(n_features, max_features, replace=False)
        X_random_subset = X_random[:, feature_indices]

        tree = build_tree(X_random_subset, y_random, max_depth=max_depth, min_samples_split=min_samples_split)
        trees.append((tree, feature_indices))

    return trees


def random_forest_predict(X, trees):
    """
    Predict using the Random Forest.

    Args:
        X (ndarray): Feature matrix.
        trees (list): List of trees and their feature subsets.

    Returns:
        ndarray: Predictions aggregated from the forest.
    """
    all_predictions = []
    for tree, features in trees:
        predictions = predict(X[:, features], tree)
        all_predictions.append(predictions)

    return np.mean(all_predictions, axis=0)


# --------------------------------------------
# Example Usage
# --------------------------------------------

X_train, y_train = make_regression(n_samples=1000, n_features=10, n_informative=10)

# Build and predict with a Random Forest
forest = build_random_forest(X_train, y_train, n_trees=5, max_depth=3, min_samples_split=2)
rf_predictions = random_forest_predict(X_train, forest)
print("Random Forest Predictions:", rf_predictions)
