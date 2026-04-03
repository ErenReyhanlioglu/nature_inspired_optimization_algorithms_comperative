import numpy as np

def sphere(x):
    """
    Sphere function: Unimodal, smooth, convex.
    Ideal for testing convergence speed.
    Global optimum: f(0,0,...,0) = 0
    """
    return np.sum(x**2, axis=1)

def ackley(x):
    """
    Ackley function: Multimodal, features a deep central pit.
    Ideal for testing basin exploitation.
    Global optimum: f(0,0,...,0) = 0
    """
    x = np.atleast_2d(x)
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=1)))
    term2 = -np.exp(np.mean(np.cos(2.0 * np.pi * x), axis=1))
    return term1 + term2 + 20 + np.exp(1)

def zakharov(x):
    """
    Zakharov function: Unimodal, non-separable, steep valleys.
    Ideal for testing high-dimensional fine-tuning and exploitation.
    Global optimum: f(0,0,...,0) = 0
    """
    x = np.atleast_2d(x)
    d = x.shape[1]
    sum1 = np.sum(x**2, axis=1)
    indices = np.arange(1, d + 1)
    sum2 = np.sum(0.5 * indices * x, axis=1)
    return sum1 + sum2**2 + sum2**4

def get_function_details(func_name):
    """
    Returns search space bounds and known global optimum coordinates for logging.
    """
    details = {
        "sphere": {"bounds": [-100.0, 100.0], "optimum": 0.0},
        "ackley": {"bounds": [-32.768, 32.768], "optimum": 0.0},
        "zakharov": {"bounds": [-5.0, 10.0], "optimum": 0.0}
    }
    return details.get(func_name.lower())