import os, sys
import re
import pandas as pd
from concurrent.futures import ProcessPoolExecutor



class FinalResult:
    def __init__(self):
        self.add_project_folder_to_pythonpath()
        self.folder = os.path.join("logs_verification")
        self.df = pd.DataFrame(columns=["file_name",
                                        "model_type", "pruning", "seed",
                                        "epsilon", "verifier", "property",
                                        "result"])


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)

    
    def main(self):
        rows = []
        with ProcessPoolExecutor() as executor:
            for row in executor.map(self.process_file_wrapper, self.iter_file_paths(), chunksize=200):
                rows.append(row)

        self.df = pd.DataFrame(rows)
        
        os.makedirs("results", exist_ok=True)
        self.df.to_csv(os.path.join("results", "raw_result.csv"), index=False)

        self.group_by_model_and_seed()
        self.df.to_csv(os.path.join("results", "result_by_model_and_seed.csv"), index=False)

        self.group_by_model()
        self.df.to_csv(os.path.join("results", "result_by_model.csv"), index=False)


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
        model_type = self.regex_helper(file_name, content, "MODEL")
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
            "model_type": model_type,
            "pruning": pruning,
            "seed": seed,
            "epsilon": epsilon,
            "verifier": verifier,
            "property": prop,
            "result": result
        }
        

    def group_by_model_and_seed(self):
        self.df = (
            self.df
            .groupby(["model_type", "pruning", "seed", "epsilon", "property"], as_index=False)
            .agg(result=("result", "max"))
        )

        self.df = (
            self.df
            .groupby(["model_type", "pruning", "seed", "epsilon"], as_index=False)
            .agg(result=("result", "sum"))
        )

        self.df = self.df.sort_values(["model_type", "epsilon", "pruning", "seed"])


    def group_by_model(self):
        self.df = (
            self.df
            .groupby(["model_type", "pruning", "epsilon"], as_index=False)
            .agg(result=("result", "sum"),
                 std=("result", "std"))
        )

        self.df = self.df.sort_values(["model_type", "epsilon", "pruning"])


    def regex_helper(self, file_name, content, header):
        pattern = rf"{header}:\s*(\S+)"
        match = re.search(pattern, content)
        if match:
            result = match.group(1)
            return result
        else:
            print(f"Error processing {header} in {file_name}")
            return None



if __name__ == "__main__":
    fr = FinalResult()
    fr.main()
