import os, sys
import re
import pandas as pd
import numpy as np
from matplotlib.patches import ConnectionPatch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from concurrent.futures import ProcessPoolExecutor



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
        df_accuracy = pd.read_csv(os.path.join("results", "MNIST_model_stats.csv"))
        df_accuracy = df_accuracy[df_accuracy["prune_type"] == "baseline"]
        df_accuracy = df_accuracy[["seed", "accuracy"]]
        df_accuracy["accuracy"] *= 100

        df_robustness = pd.read_csv(os.path.join("results", "result_by_model_and_seed.csv"))
        df_robustness = df_robustness[df_robustness["dataset"] == "MNIST"]
        df_robustness = df_robustness[df_robustness["pruning"] == "baseline"]
        df_robustness = df_robustness[df_robustness["epsilon"] == 0.007]
        df_robustness = df_robustness[["seed", "result"]]

        df = df_accuracy.merge(df_robustness, on="seed", how="inner")
        print(df)

        points_zoom = df['accuracy'].tolist()
        points_comp = df['result'].tolist()

        fig = plt.figure(figsize=(10, 8))

        # 3x3 grid: Top row has Fig 2 on the far right (1/3 width)
        # Middle and Bottom rows have Fig 1 and Fig 3 spanning full width
        gs = fig.add_gridspec(3, 3, hspace=0.6, wspace=0.1)

        ax2 = fig.add_subplot(gs[0, 2])    # Top Right (Zoom)
        ax1 = fig.add_subplot(gs[1, :])    # Middle (Original)
        ax3 = fig.add_subplot(gs[2, :])    # Bottom (Comparison)

        def style_boxed_axis(ax, data, xlim, title, color='royalblue'):
            ax.scatter(data, [0] * len(data), color=color, s=40, zorder=5)
            ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)
            ax.set_xlim(xlim)
            ax.set_ylim(-0.5, 0.5)
            ax.set_yticks([])
            ax.set_title(title, fontsize=9, fontweight='bold')
            for spine in ax.spines.values():
                spine.set_visible(True)

        # 1. Plot the data
        style_boxed_axis(ax1, points_zoom, (0, 100), "MNIST Models - Accuracy")
        style_boxed_axis(ax3, points_comp, (0, 100), "MNIST Models - Certified Robustness", color='orange')

        z_min, z_max = min(points_zoom) - 0.05, max(points_zoom) + 0.05
        style_boxed_axis(ax2, points_zoom, (z_min, z_max), "Accuracy Zoomed In")

        # 2. Draw UNIFORM lines between corresponding points (Same Index)
        line_style = dict(color="royalblue", linestyle="-", alpha=0.2, linewidth=0.8)

        for i in range(len(points_zoom)):
            # Connect Fig 1 (Middle) UP to Fig 2 (Top-Right)
            con12 = ConnectionPatch(xyA=(points_zoom[i], 0), xyB=(points_zoom[i], 0),
                                    coordsA="data", coordsB="data",
                                    axesA=ax1, axesB=ax2, **line_style)
            fig.add_artist(con12)

            # Connect Fig 1 (Middle) DOWN to Fig 3 (Bottom)
            con13 = ConnectionPatch(xyA=(points_zoom[i], 0), xyB=(points_comp[i], 0),
                                    coordsA="data", coordsB="data",
                                    axesA=ax1, axesB=ax3, **line_style)
            fig.add_artist(con13)

        plt.savefig(os.path.join("plots", "MNIST_variance.png"))
        plt.close()




if __name__ == "__main__":
    fr = FinalResult()
    fr.mnist_plotting()
