import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class PSO(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) implementation.
    Features vectorized velocity and position updates with stochastic components.
    Inherits administrative functions and success threshold from BaseOptimizer.
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs)

    def optimize(self):
        """
        Executes the PSO algorithm using vectorized operations for maximum performance.
        Returns a dictionary containing the final statistical metrics.
        """
        start_time = time.time()
        
        positions = self.initialize_population()
        velocities = np.zeros((self.pop_size, self.dim))
        
        personal_best_positions = positions.copy()
        personal_best_fitness = self.evaluate_fitness(positions)
        
        while self.fes_counter < self.max_fes and self.best_fitness > self.success_threshold:
            # Generate stochastic components (r1, r2) for the entire swarm
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            
            # Hyperparameters (w, c1, c2) are mapped from config
            # Velocity Update Rule: V(t+1) = w*V(t) + c1*r1*(PBest - X) + c2*r2*(GBest - X)
            cognitive_component = self.c1 * r1 * (personal_best_positions - positions)
            social_component = self.c2 * r2 * (self.best_position - positions)
            velocities = (self.w * velocities) + cognitive_component + social_component

            v_max = (self.bounds[1] - self.bounds[0]) * 0.5
            velocities = np.clip(velocities, -v_max, v_max)

            # Position Update Rule: X(t+1) = X(t) + V(t+1)
            positions = positions + velocities
            
            positions = self.enforce_boundaries(positions)
            
            current_fitness = self.evaluate_fitness(positions)
            
            improvement_mask = current_fitness < personal_best_fitness
            personal_best_positions[improvement_mask] = positions[improvement_mask]
            personal_best_fitness[improvement_mask] = current_fitness[improvement_mask]

        execution_time = time.time() - start_time
        
        is_successful = 1 if self.best_fitness <= self.success_threshold else 0
        
        return {
            "Algorithm_Name": "PSO",
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
                "Is_Successful": is_successful,
                "Execution_Time_Sec": execution_time
            }
        }