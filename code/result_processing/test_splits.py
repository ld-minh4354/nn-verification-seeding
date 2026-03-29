import os
import pandas as pd
import matplotlib.pyplot as plt


class TestSplits:
    def __init__(self):
        self.csv_path = os.path.join("results", "result_by_model_and_seed_two_groups.csv")


    def main(self):
        self.process_data()
        self.plotting()


    def process_data(self):
        df = pd.read_csv(self.csv_path)

        df = df[df["pruning"] == "baseline"]

        self.pivot_df = df.pivot_table(
            index=["dataset", "epsilon", "pruning", "seed"],
            columns="property_group",
            values="result"
        ).reset_index()

        correlations = (
            self.pivot_df
            .groupby(["dataset", "epsilon"])
            .apply(lambda g: g["0-49"].corr(g["50-99"]))
            .reset_index(name="correlation")
        )

        correlations.to_csv(os.path.join("results", "correlations.csv"))


    def plotting(self):
        # make output directory
        os.makedirs("plots", exist_ok=True)

        for (dataset, epsilon), group in self.pivot_df.groupby(["dataset", "epsilon"]):
            plt.figure()
            
            plt.scatter(group["0-49"], group["50-99"], s=100)
            
            plt.xlabel("Properties 1-50")
            plt.ylabel("Properties 51-100")
            
            corr = group["0-49"].corr(group["50-99"])
            plt.title(f"{dataset} | epsilon={epsilon}\nCorrelation: {corr:.3f}")
            
            filename = f"plots/{dataset}_eps_{epsilon}.png"
            plt.savefig(filename)
            plt.close()



if __name__ == "__main__":
    ts = TestSplits()
    ts.main()