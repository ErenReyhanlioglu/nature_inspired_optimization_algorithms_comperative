import numpy as np

def sphere(x):
    """
    Sphere function: Unimodal, smooth, convex.
    Ideal for testing convergence speed.
    Global optimum: f(0,0,...,0) = 0
    """
    # x: (N, D) matrix
    return np.sum(x**2, axis=1)

def rastrigin(x):
    """
    Rastrigin function: Multimodal with several local minima.
    Ideal for testing exploration and local minima escape.
    Global optimum: f(0,0,...,0) = 0
    """
    # x: (N, D) matrix
    d = x.shape[1]
    return 10 * d + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

def rosenbrock(x):
    """
    Rosenbrock function: Valley-shaped, deceptive gradients.
    Ideal for testing exploitation and directional precision.
    Global optimum: f(1,1,...,1) = 0
    """
    # x: (N, D) matrix
    # Sum of 100*(x_{i+1} - x_i^2)^2 + (x_i - 1)^2
    return np.sum(100 * (x[:, 1:] - x[:, :-1]**2)**2 + (x[:, :-1] - 1)**2, axis=1)

def get_function_details(func_name):
    """
    Returns search space bounds and known global optimum coordinates for logging.
    """
    details = {
        "sphere": {"bounds": [-100, 100], "optimum": 0},
        "rastrigin": {"bounds": [-5.12, 5.12], "optimum": 0},
        "rosenbrock": {"bounds": [-30, 30], "optimum": 1}
    }
    return details.get(func_name.lower())