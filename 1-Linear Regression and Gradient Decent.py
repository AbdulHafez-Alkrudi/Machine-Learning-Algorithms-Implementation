import numpy as np
import matplotlib as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 400.0])


def compute_cost(x, y, w, b):
    # m = x.shape[0]
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    cost = 1 / (2 * m) * cost
    return cost


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb - y[i])
    dj_dw /= m
    dj_db /= m
    return dj_dw, dj_db


def gradient_decent(x, y, w_str, b_str, alpha, num_iters):
    w = w_str
    b = b_str
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w_temp = w - alpha * dj_dw
        b_temp = b - alpha * dj_db
        w = w_temp
        b = b_temp
        if i % 1000 == 0:
            print(
                f"Iteration {i} : cost = {compute_cost(x, y, w, b)}, w = {w} , b = {b} , dj_dw = {dj_dw:0.3e} , dj_db = {dj_db:0.3e}")
    return w, b


def make_prediction(w, b, x):
    return w * x + b


init_w = 0
init_alpha = 0.9

w, b = gradient_decent(x_train, y_train, init_w, 0, init_alpha, 100)
print(f"1000 sqft house prediction {make_prediction(w, b, 1000):0.1f} Thousand dollars")
print(f"1200 sqft house prediction {make_prediction(w, b, 1200):0.1f} Thousand dollars")
print(f"2000 sqft house prediction {make_prediction(w, b, 2000):0.1f} Thousand dollars")
