import os, sys
import textwrap
import argparse



class GeneratePropertyCIFAR10:
    def __init__(self, epsilon=0.01, job_index = 0):
        self.add_project_folder_to_pythonpath()
        self.epsilon = str(epsilon)
        self.job_index = job_index

        self.model_types = ["baseline",
                            "prune_0.1", "prune_0.2", "prune_0.3", "prune_0.4",
                            "prune_0.5", "prune_0.6", "prune_0.7", "prune_0.8"]
        self.seed_values = list(range(10, 101, 10))
        self.property_values = list(range(100))
        
        os.makedirs(os.path.join("properties"), exist_ok=True)


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def generate(self, index):
        model_index = index // 1000
        seed_index = (index % 1000) // 100
        property_index = index % 100

        model = self.model_types[model_index]
        seed = self.seed_values[seed_index]
        property = self.property_values[property_index]

        self.print_info(model, seed, property)
        file_content = self.get_file_content(model, seed, property)

        file_path = os.path.join("properties", f"current_{self.job_index}.yaml")
        with open(file_path, "w") as f:
            f.write(file_content)


    def print_info(self, model, seed, property):
        print(f"DATASET: CIFAR10")
        print(f"MODEL: {model}")
        print(f"SEED: {seed}")
        print(f"PROPERTY: {property}")
        print(f"VERIFIER: abc")
        print(f"EPSILON: {self.epsilon}")


    def get_file_content(self, model, seed, property):
        return textwrap.dedent(f"""\
            model:
                name: eml_cifar10
                path: models/CIFAR10/{model}/resnet18-CIFAR10-{seed}.pth
            data:
                dataset: CIFAR
                mean: [0.4914, 0.4822, 0.4465]
                std:  [0.2471, 0.2435, 0.2616]
                start: {property}
                end: {property + 1}
            specification:
                norm: .inf
                epsilon: {self.epsilon}
        """)
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epsilon", type=float, default=0.01, help="Epsilon in verification")
    parser.add_argument("--index", type=int, default=0, help="Index of verification property")
    parser.add_argument("--job", type=int, default=0, help="Job array")
    args = parser.parse_args()

    gps = GeneratePropertyCIFAR10(epsilon=args.epsilon, job_index=args.job)
    gps.generate(index=args.index)