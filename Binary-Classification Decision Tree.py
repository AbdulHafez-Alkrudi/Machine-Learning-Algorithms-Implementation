#-----------------------------------------------
# This is the implementation of the Decision Tree if all the values are binary to solve a Binary-Classification Problem:
#-----------------------------------------------


import numpy as np
import matplotlib.pyplot as plt

X_train = np.array([[1, 1, 1],
                    [0, 0, 1],
                    [0, 1, 0],
                    [1, 0, 1],
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [1, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])


def entropy(p):
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def weighted_entropy(X, y, left_indices, right_indices):
    w_left = len(left_indices) / len(X)
    w_right = len(right_indices) / len(X)
    if len(left_indices) != 0:
        p_left = sum(y[left_indices]) / len(left_indices)
    else:
        p_left = 0
    if len(right_indices) != 0:
        p_right = sum(y[right_indices]) / len(right_indices)
    else:
        p_right = 0

    return w_left * entropy(p_left) + w_right * entropy(p_right)


# Here I'll assume that the selected feature is binary:


def split_indices(X, node_indices, feature_index):
    """
        Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have
        that feature = 1 and the right node those that have the feature = 0
        index feature = 0 => ear shape
        index feature = 1 => face shape
        index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []

    for i in node_indices:
        if X[i][feature_index] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices


def calc_information_gain(X, y, left_indices, right_indices):
    """
        Here, X has the elements in the node and y is theirs respectives classes
    """
    children_entropy = weighted_entropy(X, y, left_indices, right_indices)
    p_node = sum(y) / len(y)
    h_node = entropy(p_node)
    return h_node - children_entropy


def get_best_split(X, y, node_indices):
    best_value = 0
    best_feature = 0
    n = X.shape[1]
    # iterating over the features and try every time to split the tree and calculate the information gain for it:
    for feature in range(n):
        left_indices, right_indices = split_indices(X, node_indices, feature)
        info_gain = calc_information_gain(X, y, left_indices, right_indices)
        if info_gain > best_value:
            best_value = info_gain
            best_feature = feature
    return best_feature, best_value


def build_tree_recursive(X, y, node_indices, branch_name, max_depth, current_depth, tree):
    if len(node_indices) == 0:
        return tree
    if current_depth == max_depth:
        formatting = " " * current_depth + "-" * current_depth
        print(formatting, "%s leaf node with indices" % branch_name, node_indices)
        return
    best_feature, best_value = get_best_split(X, y, node_indices)
    formatting = "-" * current_depth
    print("%s Depth %d, %s: Split on feature: %d" % (formatting, current_depth, branch_name, best_feature))
    print(f"Current branch is : {branch_name}")
    #print(f"Best Feature is {best_feature} , Best Value is {best_value}")
    # if best_value < 0.5:
    #     return
    left_indices, right_indices = split_indices(X, node_indices, best_feature)
    print(f"Left Node has : {left_indices} , Right Node has: {right_indices}")

    tree.append((left_indices, right_indices, best_feature))

    build_tree_recursive(X, y, left_indices, "Left", max_depth, current_depth + 1, tree)
    build_tree_recursive(X, y, right_indices, "Right", max_depth, current_depth + 1, tree)

    return tree



nodes = np.arange(len(y_train))
print(nodes)
tree = []
print(build_tree_recursive(X_train, y_train, nodes, "root", 2, 0, tree))
