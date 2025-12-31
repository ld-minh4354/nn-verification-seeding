import os, sys
from itertools import product

import pandas as pd

import torch
from torchvision import datasets, transforms

from model_architecture_JPL import ResNet4



class ModelStatsJPL:
    def __init__(self):
        self.add_project_folder_to_pythonpath()
        self.device = torch.device("cuda")

        self.prune_type = ["baseline", "prune0.1", "prune0.2", "prune0.3", "prune0.4",
                           "prune0.5", "prune0.6", "prune0.7", "prune0.8"]
        self.seed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        self.df = pd.DataFrame(columns=["prune_type", "seed", "accuracy", "zero_weight_percentage"])


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def main(self):
        self.load_data()
        self.process_models()
        self.process_data()

    
    def process_data(self):
        os.makedirs("results", exist_ok=True)
        self.df.to_csv(os.path.join("results", "JPL_model_stats.csv"), index=False)

        self.df = (
            self.df
            .groupby(["prune_type"], as_index=False)
            .agg(accuracy_avg=("accuracy", "mean"),
                 accuracy_std=("accuracy", "std"))
        )
        self.df.to_csv(os.path.join("results", "JPL_model_stats_summary.csv"), index=False)


    def load_data(self):
        self.num_classes = 2

        transform_test = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.3989,), (0.1828,))
        ])

        test_dir = os.path.join("raw_datasets", "JPL_processed", "test")
        test_dataset = datasets.ImageFolder(root=test_dir, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


    def process_models(self):
        for prune_type, seed in product(self.prune_type, self.seed):
            model = self.load_model(prune_type, seed)
            accuracy = self.test_loop(model)
            percentage = self.zero_weights_percentage(model)

            self.df.loc[len(self.df)] = {
                "prune_type": prune_type,
                "seed": seed,
                "accuracy": accuracy,
                "zero_weight_percentage": percentage,
            }


    def load_model(self, prune_type, seed):
        model = ResNet4()
        model = model.to(self.device)

        state_dict = torch.load(os.path.join("models", "JPL", prune_type,
                                             f"JPL_{prune_type}_{seed}.pth"), 
                                map_location=self.device)
        model.load_state_dict(state_dict)

        return model
    

    def test_loop(self, model):
        model.eval()

        accuracy = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)

                pred = outputs.argmax(dim=1, keepdim=True)
                accuracy += pred.eq(targets.view_as(pred)).sum().item()

        accuracy = accuracy / len(self.test_loader.dataset)

        return accuracy
    

    def zero_weights_percentage(self, model):
        nonzero = 0
        total = 0
        for _, param in model.named_parameters():
            tensor = param.data
            nz = torch.count_nonzero(tensor).item()
            nonzero += nz
            total += tensor.numel()

        return (1 - nonzero / total) * 100



if __name__ == "__main__":
    stats = ModelStatsJPL()
    stats.main()