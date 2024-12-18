import copy, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print(f"X_train = \n {X_train} \nY_train = {y_train}")
print(f"X_train.shape = {X_train.shape}")
# Number of Training examples
m = X_train.shape[0]
# Number of features in each example
n = X_train.shape[1]
'''
 Remember that in Multiple Linear Regression:
 n is the number of features for each training example
 m is the number of training examples
 X is 2-Dimensional Vector
 Y is 1-Dimensional Vector
 W is 1-Dimensional Vector
 b still a single variable
'''
b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")


def predict(x, w, b):
    """
        single predict using linear regression

        Args:
          x (ndarray): Shape (n,) example with multiple features
          w (ndarray): Shape (n,) model parameters
          b (scalar):  model parameter

        Returns:
          p (scalar):  prediction
    """
    return np.dot(x, w) + b


# get a row from the training examples:
x_vec = X_train[0]
f_wb = predict(x_vec, w_init, b_init)
print(f"Prediction: {f_wb}")


def compute_cost(X, y, w, b):
    """
        compute cost
        Args:
          X (ndarray (m,n)): Data, m examples with n features
          y (ndarray (m,)) : target values
          w (ndarray (n,)) : model parameters
          b (scalar)       : model parameter

        Returns:
          cost (scalar): cost
        """
    cost = 0
    for i in range(m):
        cost += (predict(X[i], w, b) - y[i]) ** 2
    return cost / (2 * m)


# Compute and display cost using our pre-chosen optimal parameters.
cost = compute_cost(X_train, y_train, w_init, b_init)
print(f'Cost at optimal w : {cost}')


def compute_gradient(X, y, w, b):
    m, n = X.shape
    # Remember, the size of W is the same as the number of features
    dj_dw = np.zeros(n)
    dj_db = 0

    for i in range(m):
        error = predict(X[i], w, b) - y[i]
        for j in range(n):
            dj_dw[j] += error * X[i, j]
        dj_db += error

    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw


tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)
print(f'dj_db at initial w,b: {tmp_dj_db}')
print(f'dj_dw at initial w,b: \n {tmp_dj_dw}')




def gradient_decent(X, y, w_str, b_str, alpha, iters):
    """
        Performs batch gradient descent to learn w and b. Updates w and b by taking
        num_iters gradient steps with learning rate alpha

        Args:
          X (ndarray (m,n))   : Data, m examples with n features
          y (ndarray (m,))    : target values
          w_str (ndarray (n,)) : initial model parameters
          b_str (scalar)       : initial model parameter
          alpha (float)       : Learning rate
          num_iters (int)     : number of iterations to run gradient descent

        Returns:
          w (ndarray (n,)) : Updated values of parameters
          b (scalar)       : Updated value of parameter
    """
    J_history = []
    m, n = X.shape
    w = copy.deepcopy(w_str)
    b = b_str
    for i in range(iters):
        dj_db, dj_dw = compute_gradient(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i < 100000:  # prevent resource exhaustion
            J_history.append(compute_cost(X, y, w, b))
        if i % math.ceil(iters / 10) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]}   ")
    return w, b, J_history


initial_w = np.zeros_like(w_init)
initial_b = 0
iterations = 1000
alpha = 5.0e-7
w_final, b_final, J_history = gradient_decent(X_train, y_train, initial_w, initial_b, alpha, iterations)
print(f"W = {w_final}")

sns.set(style="whitegrid")
plt.figure(dpi=150)

plt.plot(J_history,  linestyle='-', color='b', label='cost')
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Cost")
plt.title("Cost changing during Gradient Decent")
plt.legend(fontsize=12)
plt.show()