import numpy as np
from tqdm import tqdm
import config
from src.benchmarks.functions import sphere, rastrigin, rosenbrock
from src.algorithms.pso import PSO
from src.algorithms.de import DE
from src.algorithms.gwo import GWO
from src.algorithms.abc import ABC
from src.algorithms.rcga import RCGA
from src.algorithms.es import ES
from src.utils.logger import save_run_log

# 1. Map strings to function and class objects
FUNC_MAP = {
    "sphere": sphere,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock
}

ALGO_MAP = {
    "PSO": PSO,
    "DE": DE,
    "GWO": GWO,
    "ABC": ABC,
    "RCGA": RCGA,
    "ES": ES
}

def run_experiment():
    """
    Executes 1620 independent optimization runs across 18 scenarios and 3 dimensions.
    Each run is logged hierarchically for statistical analysis.
    """
    # Total runs: 3 Dims * 3 Functions * 6 Algorithms * 30 Seeds = 1620
    total_runs = len(config.DIMENSIONS) * len(FUNC_MAP) * len(ALGO_MAP) * len(config.RANDOM_SEEDS) 
    
    # Initialize the master progress bar with a professional description
    pbar = tqdm(total=total_runs, desc="Meta-Heuristic Benchmark", unit="run")
    
    for dim in config.DIMENSIONS: 
        # Scientific scaling: FEs increases with dimensionality 
        max_fes = config.MAX_FES_PER_DIM[dim]
        
        for func_name, func_obj in FUNC_MAP.items(): 
            # Retrieve standard bounds for the specific topology 
            bounds = config.FUNCTION_CONFIG[func_name]["bounds"]
            
            for algo_name, algo_class in ALGO_MAP.items(): 
                # Extract hyperparameters from central config 
                hparams = config.HYPERPARAMETERS[algo_name].copy()
                
                for run_idx, seed in enumerate(config.RANDOM_SEEDS): 
                    pbar.set_description(f"DIM:{dim} | {func_name[:4].upper()} | {algo_name} | RUN:{run_idx+1}/30")
                    
                    optimizer = algo_class(
                        objective_func=func_obj,
                        bounds=bounds,
                        dim=dim,
                        pop_size=config.POP_SIZE,
                        max_fes=max_fes,
                        seed=seed,
                        **hparams
                    )
                    
                    # Execution Phase: Run the optimization logic
                    result = optimizer.optimize()
                    
                    # Update identifying metadata 
                    result["Run_ID"] = run_idx + 1
                    
                    # Hierarchical Logging: Save to Algorithm/Function/Dimension/Run_X.json
                    save_run_log(result)
                    
                    # Post-fix status: Show best fitness found in the current stochastic run
                    best_fit = result['Final_Evaluation']['Final_Best_Fitness'] 
                    pbar.set_postfix({"BestFit": f"{best_fit:.2e}"})
                    pbar.update(1)

    pbar.close()
    print("\n" + "="*50)
    print(f"[SUCCESS] 1620 Independent runs completed.")
    print(f"[INFO] Data stored in: data/raw/")
    print("="*50)

if __name__ == "__main__":
    run_experiment()