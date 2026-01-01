import os, sys
import textwrap
import argparse



class GeneratePropertyJPL:
    def __init__(self, epsilon, job_index):
        self.add_project_folder_to_pythonpath()
        self.epsilon = str(epsilon)
        self.job_index = job_index

        self.prune_types = ["baseline",
                            "prune0.1", "prune0.2", "prune0.3", "prune0.4",
                            "prune0.5", "prune0.6", "prune0.7", "prune0.8"]
        self.seed_values = list(range(10, 101, 10))
        self.property_values = list(range(100))
        
        os.makedirs(os.path.join("properties"), exist_ok=True)


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def generate(self, index):
        prune_index = index // 1000
        seed_index = (index % 1000) // 100
        property_index = index % 100

        prune = self.prune_types[prune_index]
        seed = self.seed_values[seed_index]
        property = self.property_values[property_index]

        self.print_info(prune, seed, property)
        file_content = self.get_file_content(prune, seed, property)

        file_path = os.path.join("properties", f"JPL_{self.epsilon}_{self.job_index}.yaml")
        with open(file_path, "w") as f:
            f.write(file_content)


    def print_info(self, prune, seed, property):
        print(f"DATASET: JPL")
        print(f"PRUNING: {prune}")
        print(f"SEED: {seed}")
        print(f"PROPERTY: {property}")
        print(f"VERIFIER: abc")
        print(f"EPSILON: {self.epsilon}")


    def get_file_content(self, prune, seed, property):
        return textwrap.dedent(f"""\
            model:
                name: resnet4
                path: models/JPL/{prune}/JPL_{prune}_{seed}.pth
            data:
                dataset: Customized("custom_model_data", "jpl")
                mean: [0.3989]
                std:  [0.1828]
                start: {property}
                end: {property + 1}
            specification:
                norm: .inf
                epsilon: {self.epsilon}
        """)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--index", type=int)
    parser.add_argument("--job", type=int)
    args = parser.parse_args()

    gps = GeneratePropertyJPL(epsilon=args.epsilon, job_index=args.job)
    gps.generate(index=args.index)