import numpy as np

def sphere(x, shift_vector=None):
    """
    Shifted Sphere function: Unimodal, smooth, convex.
    Ideal for testing convergence speed.
    Global optimum: f(shift_vector) = 0
    """
    x = np.atleast_2d(x)
    if shift_vector is not None:
        x = x - shift_vector
        
    return np.sum(x**2, axis=1)

def ackley(x, shift_vector=None):
    """
    Shifted Ackley function: Multimodal, features a deep central pit.
    Ideal for testing basin exploitation.
    Global optimum: f(shift_vector) = 0
    """
    x = np.atleast_2d(x)
    if shift_vector is not None:
        x = x - shift_vector
        
    term1 = -20.0 * np.exp(-0.2 * np.sqrt(np.mean(x**2, axis=1)))
    term2 = -np.exp(np.mean(np.cos(2.0 * np.pi * x), axis=1))
    return term1 + term2 + 20.0 + np.exp(1.0)

def zakharov(x, shift_vector=None):
    """
    Shifted Zakharov function: Unimodal, non-separable, steep valleys.
    Ideal for testing high-dimensional fine-tuning and exploitation.
    Global optimum: f(shift_vector) = 0
    """
    x = np.atleast_2d(x)
    if shift_vector is not None:
        x = x - shift_vector
        
    d = x.shape[1]
    sum1 = np.sum(x**2, axis=1)
    indices = np.arange(1, d + 1)
    
    # Broadcasting indices over the population matrix
    sum2 = np.sum(0.5 * indices * x, axis=1)
    
    return sum1 + sum2**2 + sum2**4

def get_function_details(func_name):
    """
    Returns the static boundaries and the theoretical optimum value.
    The spatial location of the optimum is determined dynamically by the shift_vector.
    """
    details = {
        "sphere": {"bounds": [-100.0, 100.0], "optimum": 0.0},
        "ackley": {"bounds": [-32.768, 32.768], "optimum": 0.0},
        "zakharov": {"bounds": [-5.0, 10.0], "optimum": 0.0}
    }
    return details.get(func_name.lower())