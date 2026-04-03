import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class GWO(BaseOptimizer):
    """
    Grey Wolf Optimizer (GWO) implementation.
    Mimics the leadership hierarchy and hunting mechanism of grey wolves.
    Inherits administrative functions and success threshold from BaseOptimizer.
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs)

        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')
        
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')
        
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')

    def optimize(self):
        """
        Executes the GWO hunting loop with centralized budget and threshold control.
        """
        start_time = time.time()
        positions = self.initialize_population()

        max_iterations = self.max_fes // self.pop_size
        iteration = 0

        while self.fes_counter < self.max_fes and self.best_fitness > self.success_threshold:
            fitness_values = self.evaluate_fitness(positions)
            iteration += 1
            
            # --- LEADER UPDATE LOGIC ---
            sorted_indices = np.argsort(fitness_values)
            for idx in sorted_indices:
                current_fitness = fitness_values[idx]
                
                if current_fitness < self.alpha_score:
                    # Update hierarchy: Delta <- Beta <- Alpha <- Current Best
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                    self.alpha_score, self.alpha_pos = current_fitness, positions[idx].copy()
                elif current_fitness < self.beta_score:
                    # Update hierarchy: Delta <- Beta <- Current Best
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = current_fitness, positions[idx].copy()
                elif current_fitness < self.delta_score:
                    # Update hierarchy: Delta <- Current Best
                    self.delta_score, self.delta_pos = current_fitness, positions[idx].copy()

            # --- POSITION UPDATE PHASE ---
            a = self.a_start - (iteration * ((self.a_start - self.a_end) / max_iterations))
            
            r1 = np.random.rand(3, self.pop_size, self.dim)
            r2 = np.random.rand(3, self.pop_size, self.dim)
            
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            
            D_alpha = np.abs(C[0] * self.alpha_pos - positions)
            D_beta  = np.abs(C[1] * self.beta_pos - positions)
            D_delta = np.abs(C[2] * self.delta_pos - positions)
            
            X1 = self.alpha_pos - A[0] * D_alpha
            X2 = self.beta_pos  - A[1] * D_beta
            X3 = self.delta_pos - A[2] * D_delta
            
            positions = (X1 + X2 + X3) / 3.0
            positions = self.enforce_boundaries(positions)

        execution_time = time.time() - start_time
        
        return {
            "Algorithm_Name": "GWO",
            "Function_Name": self.objective_func.__name__,
            "Dimension": self.dim,
            "Run_ID": None,
            "Random_Seed": self.seed,
            "Hyperparameters": self.hparams,
            "Convergence_Data": {
                "FEs_Milestones": self.fes_milestones,
                "Fitness_Trajectory": self.convergence_curve
            },
            "Final_Evaluation": {
                "Final_Best_Fitness": self.best_fitness,
                "Final_Best_Position": self.best_position.tolist(),
                "Is_Successful": 1 if self.best_fitness <= self.success_threshold else 0,
                "Execution_Time_Sec": execution_time
            }
        }