import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.stats import ranksums
from IPython.display import display

warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'lines.linewidth': 2.5
})

class StatisticsEngine:
    """
    Statistical processing engine for continuous optimization benchmarks.
    Designed for Jupyter Notebook cell-by-cell execution.
    """
    def __init__(self, raw_data_dir="data/raw", processed_dir="data/processed", figures_dir="outputs/figures"):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_dir = Path(processed_dir)
        self.figures_dir = Path(figures_dir)
        
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = self._load_data()

    def _load_data(self):
            """
            Loads JSON logs into a Pandas DataFrame.
            Includes positional data for spatial validation.
            Throws explicit exceptions if data directory is empty or missing.
            """
            if not self.raw_data_dir.exists():
                raise FileNotFoundError(f"Data directory does not exist: {self.raw_data_dir.resolve()}")

            json_files = list(self.raw_data_dir.rglob("*.json"))
            if not json_files:
                raise ValueError(f"No JSON files found inside: {self.raw_data_dir.resolve()}")

            records = []
            for json_path in json_files:
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    record = {
                        "Algorithm": data["Algorithm_Name"],
                        "Function": data["Function_Name"],
                        "Dimension": data["Dimension"],
                        "Run_ID": data["Run_ID"],
                        "Seed": data["Random_Seed"],
                        "Fitness": data["Final_Evaluation"]["Final_Best_Fitness"],
                        "Success": data["Final_Evaluation"]["Is_Successful"],
                        "Time_Sec": data["Final_Evaluation"]["Execution_Time_Sec"],
                        "FEs": data["Convergence_Data"]["FEs_Milestones"],
                        "Trajectory": data["Convergence_Data"]["Fitness_Trajectory"],
                        "Position": data["Final_Evaluation"]["Final_Best_Position"],
                        "Shift_Vector": data.get("Shift_Vector", [0.0] * data["Dimension"])
                    }
                    records.append(record)
                except Exception as e:
                    print(f"[ERROR] Corrupted JSON parsing failed at {json_path.name}: {str(e)}")
                    
            df = pd.DataFrame(records)
            print(f"[INFO] Successfully loaded {len(df)} records into the analytical matrix.")
            return df

    def get_dispersion_tables(self):
        """
        Test 1: Statistical Dispersion with Single-Sample Reliability Test.
        Calculates p-value for each row using a Binomial Test to prove that 
        the success rate is not due to random chance (H0: p <= 0.5).
        """
        from scipy.stats import binomtest
        
        # Base aggregation
        stats = self.df.groupby(['Dimension', 'Function', 'Algorithm']).agg(
            Best=('Fitness', 'min'),
            Worst=('Fitness', 'max'),
            Mean=('Fitness', 'mean'),
            Std=('Fitness', 'std'),
            Success_Count=('Success', 'sum') # Number of successes out of 30
        ).reset_index()
        
        # Calculate Success Rate (SR)
        stats['SR'] = (stats['Success_Count'] / 30) * 100
        
        # Statistical Significance (SR): Two-sided Binomial Test
        # H0: Success probability is exactly 0.5 (Random/chance performance)
        # H1: Success probability is NOT 0.5 (Significant non-random success or failure)
        # This confirms that both perfect success (30/30) and perfect failure (0/30) 
        # are statistically significant deviations from a random baseline.
        stats['p-value (SR)'] = stats['Success_Count'].apply(
            lambda x: binomtest(int(x), n=30, p=0.5, alternative='two-sided').pvalue
        )
        
        # Clean up for the final table
        stats.drop(columns=['Success_Count'], inplace=True)
        stats.to_csv(self.processed_dir / "statistical_dispersion.csv", index=False)
        return stats


    def plot_scalability(self):
        """
        Test 3: Scalability Matrix (Dimension vs Fitness)
        Visualizes the Curse of Dimensionality using grouped bar charts on a logarithmic scale.
        """
        stats = self.df.groupby(['Algorithm', 'Function', 'Dimension'])['Fitness'].mean().reset_index()
        
        for func in stats['Function'].unique():
            plt.figure(figsize=(10, 6))
            func_data = stats[stats['Function'] == func]
            
            sns.barplot(data=func_data, x='Dimension', y='Fitness', hue='Algorithm', edgecolor='black')
            
            plt.yscale('log')
            plt.title(f'Scalability Matrix - {func.capitalize()}')
            plt.xlabel('Problem Dimension (D)')
            plt.ylabel('Mean Fitness (Log Scale)')
            plt.grid(axis='y', ls="--", alpha=0.7)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            plt.savefig(self.figures_dir / f"scalability_bar_{func}.png", dpi=300)
            plt.show()

    def plot_convergence_curves(self):
        """
        Test 4: Mean Convergence Trajectories (1x3 Grids per Dimension)
        Generates a separate 1x3 subplot grid for each dimension to facilitate
        presentation slide integration.
        Resolves FEs length inconsistencies via linear interpolation.
        """
        dimensions = sorted(self.df['Dimension'].unique())
        functions = self.df['Function'].unique()
        
        for dim in dimensions:
            # Create a 1x3 grid for the current dimension
            fig, axes = plt.subplots(nrows=1, ncols=len(functions), figsize=(18, 6))
            
            # Ensure axes is an array even if there is only 1 function
            if len(functions) == 1:
                axes = [axes]
                
            # Stores handles and labels for a single global legend per figure
            handles, labels = [], []
            
            for j, func in enumerate(functions):
                ax = axes[j]
                subset = self.df[(self.df['Dimension'] == dim) & (self.df['Function'] == func)]
                
                if subset.empty:
                    ax.set_visible(False)
                    continue
                
                for algo in sorted(subset['Algorithm'].unique()):
                    algo_data = subset[subset['Algorithm'] == algo]
                    
                    # Find maximum FEs for interpolation grid
                    all_fes = [max(fes) for fes in algo_data['FEs']]
                    if not all_fes:
                        continue
                        
                    max_fes_val = max(all_fes)
                    common_fes = np.linspace(0, max_fes_val, 100)
                    
                    standardized_trajectories = []
                    for fes, traj in zip(algo_data['FEs'], algo_data['Trajectory']):
                        interp_traj = np.interp(common_fes, fes, traj)
                        standardized_trajectories.append(interp_traj)
                        
                    mean_trajectory = np.mean(standardized_trajectories, axis=0)
                    
                    # Plot on the specific subplot
                    line, = ax.plot(common_fes, mean_trajectory, label=algo)
                    
                    # Capture legend info only once (from the first subplot)
                    if j == 0:
                        handles.append(line)
                        labels.append(algo)
                
                # Subplot formatting
                ax.set_yscale('log')
                ax.set_title(f"{func.capitalize()} (D={dim})")
                ax.set_xlabel('Function Evaluations (FEs)')
                if j == 0:
                    ax.set_ylabel('Log(Fitness)')
                ax.grid(True, which="both", ls="--", alpha=0.5)

            # Add a single master legend at the bottom of the 1x3 figure
            fig.legend(handles, labels, loc='lower center', ncol=len(labels), bbox_to_anchor=(0.5, -0.15), fontsize=14)
            
            plt.tight_layout()
            # Save each dimension's grid independently
            plt.savefig(self.figures_dir / f"convergence_1x3_grid_D{dim}.png", dpi=300, bbox_inches='tight')
            plt.show()

    def plot_computational_cost(self):
        """
        Test 5: Computational Complexity (Execution Time)
        Automatically iterates over all tested dimensions, generates grouped bar
        charts for mean execution time, and returns the aggregated timing statistics.
        """
        all_time_stats = []
        dimensions = sorted(self.df['Dimension'].unique())
        
        for dim in dimensions:
            # Isolate data for the current dimension
            subset = self.df[self.df['Dimension'] == dim]
            
            # Calculate mean execution time per function and algorithm
            time_stats = subset.groupby(['Function', 'Algorithm'])['Time_Sec'].mean().reset_index()
            time_stats['Dimension'] = dim
            all_time_stats.append(time_stats)
            
            # Plotting operations
            plt.figure(figsize=(10, 6))
            sns.barplot(data=time_stats, x='Function', y='Time_Sec', hue='Algorithm', edgecolor='black')
            
            plt.title(f'Computational Complexity - D={dim}')
            plt.xlabel('Benchmark Function')
            plt.ylabel('Mean Execution Time (Sec)')
            plt.grid(axis='y', ls="--", alpha=0.7)
            
            # Relocate legend outside the plot to prevent data occlusion
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save and render the figure
            plt.savefig(self.figures_dir / f"computational_cost_D{dim}.png", dpi=300)
            plt.show()
            
        # Combine all dimensional timing stats into a single DataFrame
        final_time_df = pd.concat(all_time_stats, ignore_index=True)
        return final_time_df

    def validate_spatial_optimum(self):
        """
        Test 6: Spatial Validation (Distance from the Shifted Optimum)
        Automatically iterates over all tested dimensions, measures the Euclidean 
        distance between the reported best position and the actual shifted optimum,
        generates sequential plots, and returns aggregated statistics.
        """
        # Create a working copy to avoid SettingWithCopyWarning
        temp_df = self.df.copy()
        
        # Calculate Euclidean distance for all runs globally to optimize CPU usage
        temp_df['Euclidean_Error'] = temp_df.apply(
            lambda row: np.linalg.norm(np.array(row['Position']) - np.array(row['Shift_Vector'])), 
            axis=1
        )
        
        all_spatial_stats = []
        dimensions = sorted(temp_df['Dimension'].unique())
        
        for dim in dimensions:
            subset = temp_df[temp_df['Dimension'] == dim]
            
            spatial_stats = subset.groupby(['Function', 'Algorithm'])['Euclidean_Error'].mean().reset_index()
            spatial_stats['Dimension'] = dim
            all_spatial_stats.append(spatial_stats)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=spatial_stats, x='Function', y='Euclidean_Error', hue='Algorithm', edgecolor='black')
            
            plt.yscale('log')
            plt.title(f'Spatial Validation - D={dim}')
            plt.xlabel('Benchmark Function')
            plt.ylabel('Mean Euclidean Error (Log Scale)')
            plt.grid(axis='y', ls="--", alpha=0.7)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            plt.savefig(self.figures_dir / f"spatial_validation_D{dim}.png", dpi=300)
            plt.show()
            
        # Combine all dimensional stats into a single unified analytical matrix
        final_stats_df = pd.concat(all_spatial_stats, ignore_index=True)
        return final_stats_df