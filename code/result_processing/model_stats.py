import os, sys
from itertools import product

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from model_architecture import ResNet4, ResNet6, ResNet8



class ModelStatsMNIST:
    def __init__(self):
        self.add_project_folder_to_pythonpath()
        self.device = torch.device("cuda")

        self.prune_type = ["baseline", "prune_0.1", "prune_0.2", "prune_0.3", "prune_0.4",
                           "prune_0.5", "prune_0.6", "prune_0.7", "prune_0.8"]
        self.model_type = ["resnet4", "resnet6", "resnet8"]
        self.seed = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        self.df = pd.DataFrame(columns=["prune_type", "model_type", "seed", "accuracy", "zero_weight_percentage"])


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
        self.df.to_csv(os.path.join("results", "model_stats.csv"), index=False)

        self.df = (
            self.df
            .groupby(["prune_type", "model_type"], as_index=False)
            .agg(accuracy_avg=("accuracy", "mean"),
                 accuracy_std=("accuracy", "std"))
        )
        self.df.to_csv(os.path.join("results", "model_stats_summary.csv"), index=False)


    def load_data(self):
        self.num_classes = 10

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        os.makedirs("raw_datasets", exist_ok=True)
        test_dataset = datasets.MNIST(root="raw_datasets", train=False, download=False, transform=transform_test)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)


    def process_models(self):
        for prune_type, model_type, seed in product(self.prune_type, self.model_type, self.seed):
            model = self.load_model(prune_type, model_type, seed)
            accuracy = self.test_loop(model)
            percentage = self.zero_weights_percentage(model)

            self.df.loc[len(self.df)] = {
                "prune_type": prune_type,
                "model_type": model_type,
                "seed": seed,
                "accuracy": accuracy,
                "zero_weight_percentage": percentage,
            }


    def load_model(self, prune_type, model_type, seed):
        if model_type == "resnet4":
            model = ResNet4()
        elif model_type == "resnet6":
            model = ResNet6()
        elif model_type == "resnet8":
            model = ResNet8()
        
        model = model.to(self.device)

        state_dict = torch.load(os.path.join("models", prune_type, model_type,
                                             f"{model_type}-{prune_type}-{seed}.pth"), 
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
    stats = ModelStatsMNIST()
    stats.main()