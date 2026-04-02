import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class GWO(BaseOptimizer):
    """
    Grey Wolf Optimizer (GWO) implementation.
    Inherits administrative functions from BaseOptimizer.
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed):
        # Initialize the parent class (BaseOptimizer)
        # GWO doesn't have complex hyperparameter tuning outside of the linear 'a' decay
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed)
        
        # Pre-allocate leaders
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')
        
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')
        
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')

    def optimize(self):
        """
        Executes the Grey Wolf Optimizer using vectorized operations.
        Returns a dictionary containing the final statistical metrics.
        """
        start_time = time.time()
        
        # 1. Initialization phase
        positions = self.initialize_population()
        
        # 2. Main Optimization Loop
        # Loop strictly depends on the Function Evaluations limit
        while self.fes_counter < self.max_fes:
            
            # Evaluate fitness for the entire pack
            # evaluate_fitness handles FEs counting and BaseOptimizer's Global Best updates
            fitness_values = self.evaluate_fitness(positions)
            
            # Identify Alpha, Beta, and Delta wolves dynamically
            # Sort fitness values to find the top 3 distinct leaders
            sorted_indices = np.argsort(fitness_values)
            
            for idx in sorted_indices:
                current_fitness = fitness_values[idx]
                if current_fitness < self.alpha_score:
                    # Shift hierarchy down
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    self.beta_score = self.alpha_score
                    self.beta_pos = self.alpha_pos.copy()
                    # Update Alpha
                    self.alpha_score = current_fitness
                    self.alpha_pos = positions[idx].copy()
                elif current_fitness < self.beta_score:
                    self.delta_score = self.beta_score
                    self.delta_pos = self.beta_pos.copy()
                    # Update Beta
                    self.beta_score = current_fitness
                    self.beta_pos = positions[idx].copy()
                elif current_fitness < self.delta_score:
                    # Update Delta
                    self.delta_score = current_fitness
                    self.delta_pos = positions[idx].copy()

            # Calculate the linearly decreasing 'a' parameter from 2 to 0
            # Driven by FEs rather than iterations for fair comparison
            a = 2.0 - (self.fes_counter * (2.0 / self.max_fes))
            
            # Generate stochastic vectors for the entire pack simultaneously
            r1 = np.random.rand(3, self.pop_size, self.dim)
            r2 = np.random.rand(3, self.pop_size, self.dim)
            
            # Calculate A and C vectors for Alpha, Beta, and Delta
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            
            # Vectorized distance and position calculations to leaders
            # Alpha influence
            D_alpha = np.abs(C[0] * self.alpha_pos - positions)
            X1 = self.alpha_pos - A[0] * D_alpha
            
            # Beta influence
            D_beta = np.abs(C[1] * self.beta_pos - positions)
            X2 = self.beta_pos - A[1] * D_beta
            
            # Delta influence
            D_delta = np.abs(C[2] * self.delta_pos - positions)
            X3 = self.delta_pos - A[2] * D_delta
            
            # Vectorized Position Update (Omegas following the leaders)
            positions = (X1 + X2 + X3) / 3.0
            
            # Administrative checks
            positions = self.enforce_boundaries(positions)

        # 3. Finalization and Metrics Collection
        execution_time = time.time() - start_time
        
        # Define success threshold
        is_successful = 1 if self.best_fitness <= 1e-8 else 0
        
        # Structure the final output
        return {
            "Algorithm_Name": "GWO",
            "Function_Name": self.objective_func.__name__,
            "Dimension": self.dim,
            "Run_ID": None,
            "Random_Seed": self.seed,
            "Hyperparameters": {"a_start": 2.0, "a_end": 0.0},
            "Convergence_Data": {
                "FEs_Milestones": self.fes_milestones,
                "Fitness_Trajectory": self.convergence_curve
            },
            "Final_Evaluation": {
                "Final_Best_Fitness": self.best_fitness,
                "Final_Best_Position": self.best_position.tolist(),
                "Is_Successful": is_successful,
                "Execution_Time_Sec": execution_time
            }
        }