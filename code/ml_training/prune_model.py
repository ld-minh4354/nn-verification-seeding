import os, sys
import argparse
import random

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import prune
from torchvision import datasets, transforms

from model_architecture import ResNet4, ResNet6, ResNet8



class PruneMNIST:
    def __init__(self, model, seed, prune_rate):
        self.add_project_folder_to_pythonpath()
        self.device = torch.device("cuda")

        self.seed = seed
        self.set_seed(seed)

        self.prune_rate = prune_rate

        if model == 0:
            self.model = ResNet4()
            self.model_type = "resnet4"
            self.min_accuracy = 0.994
        elif model == 1:
            self.model = ResNet6()
            self.model_type = "resnet6"
            self.min_accuracy = 0.996
        elif model == 2:
            self.model = ResNet8()
            self.model_type = "resnet8"
            self.min_accuracy = 0.996
        else:
            raise ValueError("Model args passed in is not 0, 1 or 2")
        
        self.model = self.model.to(self.device)


    def add_project_folder_to_pythonpath(self):
        project_path = os.path.abspath("")
        if project_path not in sys.path:
            sys.path.append(project_path)


    def set_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def main(self):
        self.load_data()
        self.load_model()
        self.set_hyperparameters()
        self.training()
        self.save_model()


    def count_zero_weights(self):
        nonzero = 0
        total = 0
        for _, param in self.model.named_parameters():
            tensor = param.data
            nz = torch.count_nonzero(tensor)
            nonzero += nz
            total += tensor.numel()

        print(f"Non-zero weights: {nonzero}/{total}")


    def load_data(self):
        self.num_classes = 10

        transform_train = transforms.Compose([
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        os.makedirs("raw_datasets", exist_ok=True)
        train_dataset = datasets.MNIST(root="raw_datasets", train=True, download=False, transform=transform_train)
        test_dataset = datasets.MNIST(root="raw_datasets", train=False, download=False, transform=transform_test)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    
    def load_model(self):
        state_dict = torch.load(os.path.join("models", "baseline", self.model_type, f"{self.model_type}-baseline-{self.seed}.pth"), 
                                map_location=self.device)
        self.model.load_state_dict(state_dict)

        self.count_zero_weights()

        self.parameters_to_prune = []
        for _, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                self.parameters_to_prune.append((module, 'weight'))

        prune.global_unstructured(
            self.parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=self.prune_rate,
        )


    def set_hyperparameters(self):
        self.EPOCH = 100
        self.LR = 1e-3
        self.WEIGHT_DECAY = 1e-4
        self.STEP_SIZE = 30
        self.GAMMA = 0.1


    def training(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.LR, weight_decay=self.WEIGHT_DECAY)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.STEP_SIZE, gamma=self.GAMMA)

        self.criterion = nn.CrossEntropyLoss()

        print(f"Pruning {self.model_type} model with seed {self.seed} under pruning ratio {self.prune_rate}\n")

        for epoch in range(self.EPOCH):
            test_accuracy = self.train_loop(epoch)
            if test_accuracy >= self.min_accuracy:
                break

        for module, _ in self.parameters_to_prune:
            prune.remove(module, 'weight')

        self.count_zero_weights()


    def save_model(self):
        os.makedirs(os.path.join("models", f"prune_{self.prune_rate}", self.model_type), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join("models", f"prune_{self.prune_rate}", self.model_type, f"{self.model_type}-prune_{self.prune_rate}-{self.seed}.pth"))


    def train_loop(self, epoch):
        self.model.train()
        total_loss = 0

        for _, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)

            self.optimizer.zero_grad()

            loss = self.criterion(outputs, targets)
            loss.backward()
            total_loss += loss.item()

            self.optimizer.step()

        total_loss = total_loss / len(self.train_loader)
        test_loss, test_accuracy = self.test_loop()

        self.scheduler.step()
        print(f"Epoch [{epoch+1:3d}] | Train Loss: {total_loss:.4f} | Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

        return test_accuracy
    

    def test_loop(self):
        self.model.eval()

        loss = 0
        accuracy = 0

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)

                loss += self.criterion(outputs, targets).item()

                pred = outputs.argmax(dim=1, keepdim=True)
                accuracy += pred.eq(targets.view_as(pred)).sum().item()

        loss = loss / len(self.test_loader)
        accuracy = accuracy / len(self.test_loader.dataset)

        return loss, accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prune", type=int)
    args = parser.parse_args()

    pruning = PruneMNIST(model=args.model, seed=args.seed, prune_rate=args.prune / 100)
    pruning.main()
