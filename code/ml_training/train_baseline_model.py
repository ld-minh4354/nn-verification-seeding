import os, sys
import argparse
import random

import numpy as np

import torch
import torch.nn as nn
from torchvision import datasets, transforms

from model_architecture import resnet4, resnet5, resnet7
    


class TrainBaselineMNIST:
    def __init__(self, model, seed):
        self.add_project_folder_to_pythonpath()
        self.device = torch.device("cuda")

        self.seed = seed
        self.set_seed(seed)

        if model == 0:
            self.model = resnet4()
            self.model_type = "resnet4"
        elif model == 1:
            self.model = resnet5()
            self.model_type = "resnet5"
        elif model == 2:
            self.model = resnet7()
            self.model_type = "resnet7"
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
        self.set_hyperparameters()
        self.training()
        self.save_model()


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

        print(f"Start training {self.model_type} under seed {self.seed}\n")

        for epoch in range(self.EPOCH):
            test_accuracy = self.train_loop(epoch)
            if test_accuracy >= 0.993:
                break

        
    def save_model(self):
        os.makedirs(os.path.join("models", "baseline", self.model_type), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join("models", "baseline", self.model_type, f"{self.model_type}-baseline-{self.seed}.pth"))


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
    args = parser.parse_args()

    training = TrainBaselineMNIST(model=args.model, seed=args.seed)
    training.main()
