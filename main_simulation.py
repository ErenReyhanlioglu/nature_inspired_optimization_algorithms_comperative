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

# 1. Map strings to function and class objects [cite: 8, 9]
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
    Executes the full benchmark of 1620 runs with a dynamic, informative progress bar.
    Tracks Algorithm, Topology, and Dimension in real-time[cite: 187, 271].
    """
    # Total calculations: 3 Dims * 3 Funcs * 6 Algos * 30 Seeds 
    total_runs = len(config.DIMENSIONS) * len(FUNC_MAP) * len(ALGO_MAP) * len(config.RANDOM_SEEDS)
    
    # Initialize the master progress bar with empty desc to be updated dynamically
    pbar = tqdm(total=total_runs, desc="", unit="run")
    
    for dim in config.DIMENSIONS:
        # FEs limit scales with D to handle the Curse of Dimensionality 
        max_fes = config.MAX_FES_PER_DIM[dim]
        
        for func_name, func_obj in FUNC_MAP.items():
            bounds = config.FUNCTION_CONFIG[func_name]["bounds"]
            
            for algo_name, algo_class in ALGO_MAP.items():
                hparams = config.HYPERPARAMETERS[algo_name].copy()
                
                for run_idx, seed in enumerate(config.RANDOM_SEEDS):
                    current_status = f"Alg:{algo_name:4} | Topo:{func_name[:4].upper()} | Dim:{dim:2}"
                    pbar.set_description(current_status)
                    
                    # Instantiate and run the stochastic algorithm 
                    optimizer = algo_class(
                        objective_func=func_obj,
                        bounds=bounds,
                        dim=dim,
                        pop_size=config.POP_SIZE,
                        max_fes=max_fes,
                        seed=seed,
                        **hparams
                    )
                    
                    result = optimizer.optimize()
                    result["Run_ID"] = run_idx + 1 # Assign 1-30 
                    
                    # Log data to hierarchical Algorithm/Function/Dimension structure
                    save_run_log(result)
                    
                    # --- PERFORMANCE FEEDBACK ---
                    # Display the current best fitness found in the run 
                    best_fit = result['Final_Evaluation']['Final_Best_Fitness']
                    pbar.set_postfix({"BestFit": f"{best_fit:.2e}", "Run": f"{run_idx+1}/30"})
                    pbar.update(1)

    pbar.close()
    print("\n" + "="*60)
    print(f"[SUCCESS] Comparative analysis of {total_runs} runs completed.")
    print(f"[INFO] Raw data structured in: data/raw/")
    print("="*60)

if __name__ == "__main__":
    run_experiment()