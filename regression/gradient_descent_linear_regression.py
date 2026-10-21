"""Univariate linear regression trained with batch gradient descent (implemented from scratch)."""
import copy
import math

import numpy as np


def compute_cost(x, y, w, b):
    """
    Computes the cost function for linear regression.

    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities)
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model

    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0]

    total_cost = 0

    # Variable to keep track of sum of cost from each example
    cost_sum = 0

    # Loop over training examples
    for i in range(m):
        # Compute the prediction f_wb for the ith example
        f_wb = w * x[i] + b
        # Compute the cost associated with the ith example
        cost = (f_wb - y[i])**2

        # Add to sum of cost for each example
        cost_sum += cost

    # Get the total cost as the sum divided by (2*m)
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost


def compute_gradient(x, y, w, b):
    """
    Computes the gradient for linear regression
    Args:
      x (ndarray): Shape (m,) Input to the model (Population of cities)
      y (ndarray): Shape (m,) Label (Actual profits for the cities)
      w, b (scalar): Parameters of the model
    Returns
      dj_dw (scalar): The gradient of the cost w.r.t. the parameters w
      dj_db (scalar): The gradient of the cost w.r.t. the parameter b
     """

    # Number of training examples
    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    # Loop over training examples
    for i in range (m):
        # Compute the prediction f_wb for the ith example
        f_wb = w * x[i] + b
        # Compute the gradients w.r.t w and b
        dj_db += f_wb - y[i]
        dj_dw += (f_wb - y[i]) * x[i]

    dj_dw /= m
    dj_db /= m


    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      x :    (ndarray): Shape (m,)
      y :    (ndarray): Shape (m,)
      w_in, b_in : (scalar) Initial values of parameters of the model
      cost_function: function to compute cost
      gradient_function: function to compute the gradient
      alpha : (float) Learning rate
      num_iters : (int) number of iterations to run gradient descent
    Returns
      w : (ndarray): Shape (1,) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """

    # number of training examples
    m = len(x)

    # An array to store cost J and w's at each iteration — primarily for graphing later
    J_history = []
    w_history = []
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_dw, dj_db = gradient_function(x, y, w, b )

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion
            cost =  cost_function(x, y, w, b)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0:
            w_history.append(w)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")

    return w, b, J_history, w_history  # w and J/w history for graphing


if __name__ == "__main__":
    # Small synthetic example: y = 2x + 1 with noise
    rng = np.random.default_rng(42)
    x_train = rng.uniform(0, 10, 50)
    y_train = 2 * x_train + 1 + rng.normal(0, 1, 50)
    w, b, _, _ = gradient_descent(x_train, y_train, 0.0, 0.0, compute_cost, compute_gradient, 0.01, 1500)
    print(f"Learned parameters: w = {w:.3f}, b = {b:.3f}")
