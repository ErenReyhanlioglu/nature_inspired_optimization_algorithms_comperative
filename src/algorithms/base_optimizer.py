import numpy as np
from abc import ABC, abstractmethod
import time

class BaseOptimizer(ABC):
    """
    Abstract Base Class for meta-heuristic continuous space optimization algorithms.
    Handles administrative tasks: boundary enforcement, population initialization,
    Function Evaluations (FEs) counting, and statistical trajectory logging.
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed):
        self.objective_func = objective_func
        self.bounds = bounds
        self.dim = dim
        self.pop_size = pop_size
        self.max_fes = max_fes
        self.seed = seed
        
        # Operational states
        self.fes_counter = 0
        self.best_fitness = float('inf')
        self.best_position = np.zeros(dim)
        
        # Data logging structures for Convergence Curve and Final Output
        self.convergence_curve = []
        self.fes_milestones = []
        
        # Ensure reproducibility by fixing the stochastic seed per run
        np.random.seed(self.seed)

    def initialize_population(self):
        """
        Generates the initial population uniformly distributed within the search space bounds.
        Returns an N x D numpy array.
        """
        lower_bound, upper_bound = self.bounds
        return np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))

    def enforce_boundaries(self, positions):
        """
        Clamps the agents' coordinates strictly within the allowed search space bounds.
        Prevents mathematical divergence during aggressive exploration phases.
        """
        lower_bound, upper_bound = self.bounds
        return np.clip(positions, lower_bound, upper_bound)

    def evaluate_fitness(self, positions):
        """
        Calculates the objective function values for the entire population.
        Updates the global best parameters and strictly increments the FEs counter.
        Logs the trajectory data dynamically to prevent memory bottlenecks.
        """
        # Vectorized fitness evaluation
        fitness_values = self.objective_func(positions)
        self.fes_counter += len(positions)
        
        # Identify the current best agent
        current_best_idx = np.argmin(fitness_values)
        current_best_val = fitness_values[current_best_idx]
        
        # Update global best if a superior solution is found
        if current_best_val < self.best_fitness:
            self.best_fitness = current_best_val
            self.best_position = positions[current_best_idx].copy()
            
        # Trajectory logging (Records the absolute global best found so far)
        self.fes_milestones.append(self.fes_counter)
        self.convergence_curve.append(self.best_fitness)
        
        return fitness_values

    @abstractmethod
    def optimize(self):
        """
        Abstract method. Every specific algorithm inheriting from BaseOptimizer 
        MUST implement its own topological search mechanics here.
        Must return a structured dictionary containing final evaluation metrics.
        """
        pass