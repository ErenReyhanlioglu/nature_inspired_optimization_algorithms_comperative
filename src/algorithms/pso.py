import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class PSO(BaseOptimizer):
    """
    Particle Swarm Optimization (PSO) implementation.
    Inherits administrative functions from BaseOptimizer.
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs)

    def optimize(self):
        """
        Executes the PSO algorithm using vectorized operations.
        Returns a dictionary containing the final statistical metrics.
        """
        start_time = time.time()
        
        positions = self.initialize_population()
        velocities = np.zeros((self.pop_size, self.dim))
        
        # Initialize personal bests (PBest)
        personal_best_positions = positions.copy()
        # evaluate_fitness handles FEs counting and Global Best (GBest) updates
        personal_best_fitness = self.evaluate_fitness(positions)
        
        # Main Optimization Loop
        # Loop strictly depends on the Function Evaluations limit, not iterations
        while self.fes_counter < self.max_fes and self.best_fitness > 1e-8:
            # Generate stochastic components for the entire swarm simultaneously
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            
            # self.w, self.c1, self.c2 are automatically available via setattr in BaseOptimizer
            cognitive_component = self.c1 * r1 * (personal_best_positions - positions)
            social_component = self.c2 * r2 * (self.best_position - positions)
            velocities = (self.w * velocities) + cognitive_component + social_component
            
            # Vectorized Position Update
            positions = positions + velocities
            
            # Administrative checks (Handled by BaseOptimizer)
            positions = self.enforce_boundaries(positions)
            
            # Evaluate new positions
            # Note: evaluate_fitness automatically updates self.best_fitness and self.best_position
            current_fitness = self.evaluate_fitness(positions)
            
            # Update Personal Bests using boolean masking (Highly efficient)
            improvement_mask = current_fitness < personal_best_fitness
            personal_best_positions[improvement_mask] = positions[improvement_mask]
            personal_best_fitness[improvement_mask] = current_fitness[improvement_mask]

        # Finalization and Metrics Collection
        execution_time = time.time() - start_time
        
        # Define success threshold (epsilon = 1e-8)
        is_successful = 1 if self.best_fitness <= 1e-8 else 0
        
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