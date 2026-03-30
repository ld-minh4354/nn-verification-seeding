import os, sys
import re
import pandas as pd
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from concurrent.futures import ProcessPoolExecutor
from matplotlib.ticker import FixedLocator, ScalarFormatter



class FinalResult:
    def __init__(self):
        self.add_project_folder_to_pythonpath()
        self.folder = os.path.join("logs_verification")
        self.df = pd.DataFrame(columns=["file_name",
                                        "dataset", "pruning", "seed",
                                        "epsilon", "verifier", "property",
                                        "result"])


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)

    
    def main(self):
        # rows = []
        # with ProcessPoolExecutor() as executor:
        #     for row in executor.map(self.process_file_wrapper, self.iter_file_paths(), chunksize=200):
        #         rows.append(row)

        # self.df = pd.DataFrame(rows)
        
        # os.makedirs("results", exist_ok=True)
        # self.df.to_csv(os.path.join("results", "raw_result.csv"), index=False)

        self.df = pd.read_csv(os.path.join("results", "raw_result.csv"))

        df = self.group_by_model_and_seed_two_groups()
        df.to_csv(os.path.join("results", "result_by_model_and_seed_two_groups.csv"), index=False)

        self.group_by_model_and_seed()
        self.df.to_csv(os.path.join("results", "result_by_model_and_seed.csv"), index=False)

        self.group_by_model()
        self.df.to_csv(os.path.join("results", "result_by_model.csv"), index=False)

        self.graph_plotting()


    def iter_file_paths(self):
        for entry in os.scandir(self.folder):
            if entry.is_file():
                yield entry.path


    def process_file_wrapper(self, file_path):
        file_name = os.path.basename(file_path)
        with open(file_path, "r") as f:
            content = f.read()
        return self.process_file(file_name, content)
    

    def process_file(self, file_name, content):
        dataset = self.regex_helper(file_name, content, "DATASET")
        pruning = self.regex_helper(file_name, content, "PRUNING")
        seed = self.regex_helper(file_name, content, "SEED")
        prop = self.regex_helper(file_name, content, "PROPERTY")
        verifier = self.regex_helper(file_name, content, "VERIFIER")
        epsilon = self.regex_helper(file_name, content, "EPSILON")

        if "total verified (safe/unsat): 1" in content:
            result = 1
        else:
            result = 0

        return {
            "file_name": file_name,
            "dataset": dataset,
            "pruning": pruning,
            "seed": seed,
            "epsilon": epsilon,
            "verifier": verifier,
            "property": prop,
            "result": result
        }
    

    def group_by_model_and_seed_two_groups(self):
        df = self.df
        df["property_group"] = (self.df["property"] > 50).map({False: "0-49", True: "50-99"})

        df = (
            df
            .groupby(["dataset", "pruning", "seed", "epsilon", "property_group"], as_index=False)
            .agg(result=("result", "sum"))
        )

        return df
        

    def group_by_model_and_seed(self):
        self.df = (
            self.df
            .groupby(["dataset", "pruning", "seed", "epsilon", "property"], as_index=False)
            .agg(result=("result", "max"))
        )

        self.df = (
            self.df
            .groupby(["dataset", "pruning", "seed", "epsilon"], as_index=False)
            .agg(result=("result", "sum"))
        )

        self.df = self.df.sort_values(["dataset", "epsilon", "pruning", "seed"])


    def group_by_model(self):
        self.df = (
            self.df
            .groupby(["dataset", "pruning", "epsilon"], as_index=False)
            .agg(result=("result", "mean"),
                 std=("result", "std"))
        )

        self.df = self.df.sort_values(["dataset", "epsilon", "pruning"])


    def regex_helper(self, file_name, content, header):
        pattern = rf"{header}:\s*(\S+)"
        match = re.search(pattern, content)
        if match:
            result = match.group(1)
            return result
        else:
            print(f"Error processing {header} in {file_name}")
            return None
        


    def mnist_plotting(self):
        # Load and prepare data
        df_accuracy = pd.read_csv(os.path.join("results", "MNIST_model_stats.csv"))
        df_accuracy = df_accuracy[df_accuracy["prune_type"] == "baseline"]
        df_accuracy["accuracy"] *= 100

        df_robustness = pd.read_csv(os.path.join("results", "result_by_model_and_seed.csv"))
        df_robustness = df_robustness[df_robustness["dataset"] == "MNIST"]
        df_robustness = df_robustness[df_robustness["pruning"] == "baseline"]
        df_robustness = df_robustness[df_robustness["epsilon"] == 0.007]

        df = df_accuracy.merge(df_robustness[["seed", "result"]], on="seed", how="inner")
        
        points_acc = df['accuracy'].tolist()
        points_rob = df['result'].tolist()
        
        # Fixed Standard Deviation Values
        std_acc = 0.057
        std_rob = 28.8

        # 1. Compact Figure Setup
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 2.8))
        plt.subplots_adjust(hspace=1.0)
        
        def style_boxed_axis(ax, data, std_val, label_dist, title, color='royalblue'):
            ax.scatter(data, [0]*len(data), color=color, s=35, zorder=5, alpha=0.6, edgecolors='white', linewidth=0.5)
            ax.axhline(0, color='black', linewidth=0.8, alpha=0.2)
            
            # 2. Custom Scale: Logarithmic on the "Distance to 100"
            # This makes 10-90, 90-99, and 99-99.9 take up equal visual horizontal space.
            # Forward transform: -log10(100 - x); Reverse transform: 100 - 10^-x
            ax.set_xscale('function', functions=(
                lambda x: -np.log10(np.maximum(100.1 - x, 1e-5)), 
                lambda x: 100.1 - 10**(-x)
            ))
            
            # 3. Set Range 1 to 99.9
            ax.set_xlim(1, 99.9) 
            
            # 4. Tick Labels with % units
            tick_vals = [10, 50, 90, 95, 99, 99.9]
            tick_labels = ['10%', '50%', '90%', '95%', '99%', '99.9%']
            ax.xaxis.set_major_locator(FixedLocator(tick_vals))
            ax.set_xticklabels(tick_labels, fontsize=8)

            # 5. Vertical compression
            ax.set_ylim(-0.05, 0.05)
            ax.set_yticks([])
            
            ax.set_ylabel(title, fontsize=8, fontweight='bold', rotation=0, 
                        labelpad=45, va='center', ha='right')
            
            # 6. Standard Deviation notation (Fixed Values + Units)
            ax.text(label_dist, 0.65, f"$\sigma$ = {std_val}%", transform=ax.transAxes, 
                    fontsize=8, fontweight='bold', color=color, ha='right')
            
            for spine in ax.spines.values():
                spine.set_visible(True)

        # Plotting
        style_boxed_axis(ax1, points_acc, std_acc, 0.86, "MNIST\nAccuracy")
        style_boxed_axis(ax2, points_rob, std_rob, 0.23, "MNIST\nCertified\nRobustness", color='orange')

        plt.savefig(os.path.join("plots", "MNIST_variance.png"), bbox_inches='tight', dpi=300)
        plt.close()


if __name__ == "__main__":
    fr = FinalResult()
    fr.mnist_plotting()
