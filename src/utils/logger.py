import json
import os

def save_run_log(data, base_path="data/raw"):
    """
    Saves a single optimization run result to a hierarchical directory structure.
    Structure: base_path / Algorithm / Function / Dimension / Run_ID.json
    """
    # Example: data/raw/PSO/sphere/D30/
    folder_path = os.path.join(
        base_path, 
        data['Algorithm_Name'], 
        data['Function_Name'], 
        f"D{data['Dimension']}"
    )
    
    os.makedirs(folder_path, exist_ok=True)
    
    file_name = f"Run_{data['Run_ID']}.json"
    full_path = os.path.join(folder_path, file_name)
    
    with open(full_path, 'w') as f:
        json.dump(data, f, indent=4)

def check_existing_run(algo, func, dim, run_id, base_path="data/raw"):
    """Checks if the specific run file exists in the hierarchy for crash-recovery."""
    file_path = os.path.join(base_path, algo, func, f"D{dim}", f"Run_{run_id}.json")
    return os.path.exists(file_path)