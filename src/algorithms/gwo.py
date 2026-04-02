import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class GWO(BaseOptimizer):
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs)
        
        # Pre-allocate leaders
        self.alpha_pos = np.zeros(self.dim)
        self.alpha_score = float('inf')
        self.beta_pos = np.zeros(self.dim)
        self.beta_score = float('inf')
        self.delta_pos = np.zeros(self.dim)
        self.delta_score = float('inf')

    def optimize(self):
        start_time = time.time()
        positions = self.initialize_population()
        
        while self.fes_counter < self.max_fes and self.best_fitness > 1e-8:
            fitness_values = self.evaluate_fitness(positions)
            
            # Leader update logic
            sorted_indices = np.argsort(fitness_values)
            for idx in sorted_indices:
                current_fitness = fitness_values[idx]
                if current_fitness < self.alpha_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = self.alpha_score, self.alpha_pos.copy()
                    self.alpha_score, self.alpha_pos = current_fitness, positions[idx].copy()
                elif current_fitness < self.beta_score:
                    self.delta_score, self.delta_pos = self.beta_score, self.beta_pos.copy()
                    self.beta_score, self.beta_pos = current_fitness, positions[idx].copy()
                elif current_fitness < self.delta_score:
                    self.delta_score, self.delta_pos = current_fitness, positions[idx].copy()

            # Updated dynamic 'a' calculation
            # a decreases from a_start to a_end based on FEs (attributes dynamically mapped)
            a = self.a_start - (self.fes_counter * ((self.a_start - self.a_end) / self.max_fes))
            
            r1 = np.random.rand(3, self.pop_size, self.dim)
            r2 = np.random.rand(3, self.pop_size, self.dim)
            A = 2.0 * a * r1 - a
            C = 2.0 * r2
            
            # Position updates
            X1 = self.alpha_pos - A[0] * np.abs(C[0] * self.alpha_pos - positions)
            X2 = self.beta_pos - A[1] * np.abs(C[1] * self.beta_pos - positions)
            X3 = self.delta_pos - A[2] * np.abs(C[2] * self.delta_pos - positions)
            
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
                "Is_Successful": 1 if self.best_fitness <= 1e-8 else 0,
                "Execution_Time_Sec": execution_time
            }
        }