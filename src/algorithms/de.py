import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class DE(BaseOptimizer):
    """
    Differential Evolution (DE) implementation using the DE/rand/1/bin strategy.
    Inherits administrative functions from BaseOptimizer.
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs)

    def optimize(self):
        """
        Executes the Differential Evolution algorithm using vectorized operations.
        Returns a dictionary containing the final statistical metrics.
        """
        start_time = time.time()
        
        positions = self.initialize_population()
        
        fitness_values = self.evaluate_fitness(positions)
        
        population_indices = np.arange(self.pop_size)
        r1 = np.empty(self.pop_size, dtype=int)
        r2 = np.empty(self.pop_size, dtype=int)
        r3 = np.empty(self.pop_size, dtype=int)

        while self.fes_counter < self.max_fes and self.best_fitness > self.success_threshold:
            
            # Select 3 mutually exclusive random indices (r1, r2, r3) for each target vector (i)
            for i in range(self.pop_size):
                available_indices = np.delete(population_indices, i)
                selected = np.random.choice(available_indices, 3, replace=False)
                r1[i], r2[i], r3[i] = selected[0], selected[1], selected[2]

            # Vectorized Mutation: V_i = X_r1 + F * (X_r2 - X_r3)
            mutant_vectors = positions[r1] + self.F * (positions[r2] - positions[r3])
            mutant_vectors = self.enforce_boundaries(mutant_vectors)

            cross_points = np.random.rand(self.pop_size, self.dim) < self.CR
            
            j_rand = np.random.randint(0, self.dim, self.pop_size)
            cross_points[np.arange(self.pop_size), j_rand] = True

            trial_vectors = np.where(cross_points, mutant_vectors, positions)

            trial_fitness = self.evaluate_fitness(trial_vectors)

            improvement_mask = trial_fitness <= fitness_values
            positions[improvement_mask] = trial_vectors[improvement_mask]
            fitness_values[improvement_mask] = trial_fitness[improvement_mask]

        execution_time = time.time() - start_time
        
        is_successful = 1 if self.best_fitness <= self.success_threshold else 0
        
        return {
            "Algorithm_Name": "DE",
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