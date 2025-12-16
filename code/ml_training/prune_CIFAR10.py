import os, sys
import argparse
import random

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils import prune
from torchvision import datasets, transforms

from resnet18_CIFAR10 import BasicBlock, ResNet18



class PruneCIFAR10:
    def __init__(self, seed = 10, prune_rate = 0.1):
        self.add_project_folder_to_pythonpath()
        self.seed = seed
        self.set_seed(seed)
        self.prune_rate = prune_rate
        self.device = torch.device("cuda")


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


    def load_data(self):
        self.num_classes = 10

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

        os.makedirs("raw_datasets", exist_ok=True)
        train_dataset = datasets.CIFAR10(root="raw_datasets", train=True, download=False, transform=transform_train)
        test_dataset = datasets.CIFAR10(root="raw_datasets", train=False, download=False, transform=transform_test)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

    
    def load_model(self):
        self.model = ResNet18(BasicBlock, [2, 2, 2, 2], in_planes=16)
        self.model = self.model.to(self.device)

        state_dict = torch.load(os.path.join("models", "CIFAR10", "baseline", f"resnet18-CIFAR10-{self.seed}.pth"), 
                                map_location=self.device)
        self.model.load_state_dict(state_dict)

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

        print(f"Pruning ResNet18 for CIFAR10 under seed {self.seed}\n")

        for epoch in range(self.EPOCH):
            test_accuracy = self.train_loop(epoch)
            if test_accuracy >= 0.88:
                break

        for module, _ in self.parameters_to_prune:
            prune.remove(module, 'weight')


    def save_model(self):
        os.makedirs(os.path.join("models", "CIFAR10", f"prune_{self.prune_rate}"), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join("models", "CIFAR10", f"prune_{self.prune_rate}", f"resnet18-CIFAR10-{self.seed}.pth"))


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
    parser.add_argument("--seed", type=int, default=10, help="Random seed for training")
    parser.add_argument("--prune", type=int, default=10, help="Prune percentage")
    args = parser.parse_args()

    pruning = PruneCIFAR10(seed=args.seed, prune_rate=args.prune / 100)
    pruning.main()
