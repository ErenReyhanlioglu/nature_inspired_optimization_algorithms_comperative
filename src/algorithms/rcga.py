import numpy as np
import time
from src.algorithms.base_optimizer import BaseOptimizer

class RCGA(BaseOptimizer):
    """
    Real-Coded Genetic Algorithm (RCGA) implementation.
    Uses Arithmetic Crossover for exploration and Gaussian Mutation for exploitation.
    Inherits administrative functions from BaseOptimizer.
    """
    def __init__(self, objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs):
        super().__init__(objective_func, bounds, dim, pop_size, max_fes, seed, **kwargs)

    def tournament_selection(self, population, fitness):
        """
        Selects a single parent using tournament selection logic.
        self.tournament_size is dynamically mapped from kwargs in BaseOptimizer.
        """
        indices = np.random.choice(len(population), self.tournament_size, replace=False)
        tournament_fitness = fitness[indices]
        winner_idx = indices[np.argmin(tournament_fitness)]
        return population[winner_idx].copy()

    def optimize(self):
        start_time = time.time()
        
        # 1. Initialization
        population = self.initialize_population()
        fitness = self.evaluate_fitness(population)
        
        while self.fes_counter < self.max_fes and self.best_fitness > 1e-8:
            new_population = []
            
            # 2. Reproduction Loop
            while len(new_population) < self.pop_size:
                # Selection
                parent1 = self.tournament_selection(population, fitness)
                parent2 = self.tournament_selection(population, fitness)
                
                # Crossover (Arithmetic Crossover)
                # self.pc is automatically mapped from config
                if np.random.rand() < self.pc:
                    alpha = np.random.rand()
                    offspring1 = alpha * parent1 + (1 - alpha) * parent2
                    offspring2 = alpha * parent2 + (1 - alpha) * parent1
                else:
                    offspring1, offspring2 = parent1.copy(), parent2.copy()
                
                # Mutation (Gaussian Mutation)
                # self.pm is automatically mapped from config
                for offspring in [offspring1, offspring2]:
                    if np.random.rand() < self.pm:
                        # Mutation strength scales with search space width
                        scale = (self.bounds[1] - self.bounds[0]) * 0.01
                        mutation_noise = np.random.normal(0, scale, self.dim)
                        offspring += mutation_noise
                    
                    new_population.append(self.enforce_boundaries(np.atleast_2d(offspring))[0])
            
            # 3. Evaluation and Replacement
            new_population = np.array(new_population[:self.pop_size])
            new_fitness = self.evaluate_fitness(new_population)
            
            # Generational replacement (with simple elitism check)
            population = new_population
            fitness = new_fitness

        execution_time = time.time() - start_time
        is_successful = 1 if self.best_fitness <= 1e-8 else 0
        
        return {
            "Algorithm_Name": "RCGA",
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