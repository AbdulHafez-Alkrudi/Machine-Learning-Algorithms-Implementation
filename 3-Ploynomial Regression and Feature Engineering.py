import copy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(precision=2)  # reduced display precision on numpy arrays


def predict(x, w, b):
    return np.dot(x, w) + b


def compute_cost(X, y, w, b):
    cost = 0
    m, n = X.shape
    for i in range(m):
        cost += (predict(X[i], w, b) - y[i]) ** 2
    return cost / (2 * m)


def compute_derivatives(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        error = predict(X[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error
    return dj_dw / m, dj_db / m


def gradient_decent(X, y, w_init, b_init, alpha, iters):
    w = copy.deepcopy(w_init)
    b = b_init
    J_history = []
    for i in range(iters):
        dj_dw, dj_db = compute_derivatives(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J_history.append(compute_cost(X, y, w, b))
        if i % (iters/10) == 0:
            print(f"Iteration             {i}: Cost = {compute_cost(X, y, w, b)}")
    return w, b, J_history


def zscore_normalize_features(X, rtn_ms=False):
    """
    returns z-score normalized X by column
    Args:
      X : (numpy array (m,n))
    Returns
      X_norm: (numpy array (m,n)) input normalized by column
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    if rtn_ms:
        return X_norm, mu, sigma
    else:
        return X_norm


def print_res(X, y, title, iters, alpha):
    w_final, b_final, J_history = gradient_decent(X, y, np.zeros(X.shape[1]), 0, alpha, iters)

    plt.scatter(x_train, y, label="Actual Values", marker='x', color='r')
    plt.plot(x_train, np.dot(X, w_final) + b_final, color='b', label="Predicted Values")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.legend()
    plt.show()
    return w_final, b_final


x_train = np.arange(0, 20, 1)
y_train = 1 + x_train ** 2

# Finding the Best-fit-line:
# first I need to convert the training set to a two-dimensional array

X = x_train.reshape(-1, 1)
print_res(X, y_train, "The Best Fit Line", 10000, 1e-2)
#----------------------------------------------------------------------
# Finding the Best Fit Curve by changing every feature x -> x^2
X = x_train ** 2
X = X.reshape(-1, 1)
w, b = print_res(X, y_train, "The Best Fit Curve", 10000,1e-5)
print(f"cost = {compute_cost(X, y_train, w, b)}")

#----------------------------------------------
# trying out w0(x0)^1 + w1(x1)^2 + w2(x2)^3 + b:
X = np.c_[x_train, x_train ** 2, x_train ** 3]
y = x_train ** 2
print(f"X shape = {X.shape}\nX = {X}\ny = {y}")

w, b = print_res(X, y, "The Best Fit curve with X = [x  x^2  x^3]", 10000, 1e-7)
print(f"w = {w}\nb = {b}")
# Alternative way to know which power of x to choose:
'''
here I just tried to scatter the points as (X[:, i], y) ...
the columns represent x, x^2, x^3, and y = x^2
so I plotted them and the best one gonna look like a line
'''


X = np.c_[x_train, x_train ** 2, x_train ** 3]
X_features = ['x', 'x^2', 'x^3']
y = x_train ** 2

fig, ax = plt.subplots(1, 3, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X[:,i], y)
    ax[i].set_xlabel(X_features[i])

ax[0].set_ylabel('Y')
plt.show()

#Scaling the data using Z function:
X = np.c_[x_train, x_train**2, x_train**3]
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X, axis=0)}")

X = zscore_normalize_features(X)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X, axis=0)}")

#Trying more aggressive function with scaling:
x = np.arange(0,20,1)
y = np.cos(x/2)

X = np.c_[x, x**2, x**3,x**4, x**5, x**6, x**7, x**8, x**9, x**10, x**11, x**12, x**13]
X = zscore_normalize_features(X)
w, b = print_res(X, y,  "Y = cos(x/2)", 1000000, 1e-1)


