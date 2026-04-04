import argparse
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import config
from src.benchmarks.functions import sphere, ackley, zakharov
from src.algorithms.pso import PSO
from src.algorithms.de import DE
from src.algorithms.gwo import GWO
from src.algorithms.abc import ABC
from src.algorithms.es import ES
from src.utils.logger import save_run_log, check_existing_run

FUNC_MAP = {
    "sphere": sphere,
    "ackley": ackley,
    "zakharov": zakharov
}

ALGO_MAP = {
    "PSO": PSO,
    "DE": DE,
    "GWO": GWO,
    "ABC": ABC,
    "ES": ES
}

def load_single_run_result(algo, func, dim, run_id):
    """Reads the result of a specific run from disk for statistics."""
    path = Path(f"data/raw/{algo}/{func}/D{dim}/Run_{run_id}.json")
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except:
        return None

def get_dim_summary(dim, target_algos):
    """Prepares general summary tables for the scanned dimension."""
    records = []
    base_path = Path("data/raw")
    for json_path in base_path.rglob(f"D{dim}/*.json"):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            algo_name = data["Algorithm_Name"]
            if target_algos and algo_name not in target_algos:
                continue

            records.append({
                "Algorithm": algo_name,
                "Function": data["Function_Name"],
                "Fitness": data["Final_Evaluation"]["Final_Best_Fitness"],
                "Success": data["Final_Evaluation"]["Is_Successful"]
            })
        except: continue
        
    if not records: return None
    df = pd.DataFrame(records)
    summary = df.groupby(['Algorithm', 'Function']).agg(
        Mean_Fitness=('Fitness', 'mean'),
        SR=('Success', lambda x: x.mean() * 100)
    ).reset_index()
    return summary.pivot(index='Algorithm', columns='Function', values='Mean_Fitness'), \
           summary.pivot(index='Algorithm', columns='Function', values='SR')

def make_shifted_objective(base_func, shift_vector):
    """
    Creates a wrapper function to inject the shift_vector dynamically
    while preserving the original function's name for logging.
    """
    def shifted_func(x, **kwargs):
        return base_func(x, shift_vector=shift_vector, **kwargs)
    shifted_func.__name__ = base_func.__name__
    return shifted_func

def run_experiment(target_algos=None):
    # Filter ALGO_MAP based on CLI arguments
    algos_to_run = {}
    if target_algos:
        for algo in target_algos:
            if algo.upper() in ALGO_MAP:
                algos_to_run[algo.upper()] = ALGO_MAP[algo.upper()]
            else:
                print(f"[WARNING] Algorithm '{algo}' is not recognized. Skipping.")
    else:
        algos_to_run = ALGO_MAP

    if not algos_to_run:
        print("[ERROR] No valid algorithms selected for execution. Aborting.")
        return

    total_runs = len(config.DIMENSIONS) * len(FUNC_MAP) * len(algos_to_run) * len(config.RANDOM_SEEDS)
    pbar = tqdm(total=total_runs, desc="SIMULATION", unit="run")
    
    for dim in config.DIMENSIONS:
        max_fes = config.MAX_FES_PER_DIM[dim]
        dynamic_pop_size = dim * config.POP_SIZE_MULTIPLIER
        
        for func_name, func_obj in FUNC_MAP.items():
            bounds = config.FUNCTION_CONFIG[func_name]["bounds"]
            
            for algo_name, algo_class in algos_to_run.items():
                hparams = config.HYPERPARAMETERS[algo_name].copy()
                block_results = []
                
                for run_idx, seed in enumerate(config.RANDOM_SEEDS):
                    run_id = run_idx + 1
                    
                    if check_existing_run(algo_name, func_name, dim, run_id):
                        data = load_single_run_result(algo_name, func_name, dim, run_id)
                        if data: block_results.append(data)
                        pbar.update(1)
                        continue

                    current_status = f"D{dim} | {algo_name:4} | {func_name[:4].upper()}"
                    pbar.set_description(current_status)

                    # --- FUNCTION SHIFTING LOGIC ---
                    # Use a local RandomState to guarantee fair comparison across algorithms
                    rng = np.random.RandomState(seed)
                    
                    # Calculate safe boundaries (20% margin from the edges)
                    margin = (bounds[1] - bounds[0]) * 0.2
                    lower_target = bounds[0] + margin
                    upper_target = bounds[1] - margin
                    
                    # Generate the specific optimum coordinates for this run
                    target_shift = rng.uniform(lower_target, upper_target, dim)
                    shifted_objective = make_shifted_objective(func_obj, target_shift)

                    optimizer = algo_class(
                        objective_func=shifted_objective, bounds=bounds, dim=dim,
                        pop_size=dynamic_pop_size, max_fes=max_fes, seed=seed,
                        success_threshold=config.SUCCESS_THRESHOLD, **hparams
                    )
                    
                    result = optimizer.optimize()
                    result["Run_ID"] = run_id
                    
                    # Record the shifted optimum coordinates for post-analysis verification
                    result["Shift_Vector"] = target_shift.tolist()
                    
                    save_run_log(result)
                    block_results.append(result)
                    
                    pbar.set_postfix({"Fit": f"{result['Final_Evaluation']['Final_Best_Fitness']:.1e}"})
                    pbar.update(1)
                
                if block_results:
                    fits = [r['Final_Evaluation']['Final_Best_Fitness'] for r in block_results]
                    srs = [r['Final_Evaluation']['Is_Successful'] for r in block_results]
                    m_fit = np.mean(fits)
                    m_sr = np.mean(srs) * 100
                    pbar.write(f"[OK] {algo_name:4} - {func_name:10} (D{dim}) | Mean: {m_fit:.2e} | SR: %{m_sr:.1f}")

        pbar.write(f"\n{'='*25} D{dim} GENERAL REPORT {'='*25}")
        res = get_dim_summary(dim, algos_to_run.keys())
        if res:
            fit_t, sr_t = res
            pbar.write(f"\n[MEAN FITNESS] - D{dim}\n{fit_t.to_markdown(floatfmt='.2e')}")
            pbar.write(f"\n[SUCCESS RATE %] - D{dim}\n{sr_t.to_markdown(floatfmt='.1f')}")
        pbar.write(f"{'='*75}\n")

    pbar.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Metaheuristic Optimization Simulation Framework")
    parser.add_argument(
        '--algo', 
        type=str, 
        nargs='+', 
        help='Specify one or more algorithms to run (e.g., --algo ABC PSO). Runs all if left empty.'
    )
    
    args = parser.parse_args()
    run_experiment(target_algos=args.algo)