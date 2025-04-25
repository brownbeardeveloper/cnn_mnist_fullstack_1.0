import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import os
import numpy as np
import random
from tqdm import tqdm


# Set random seeds for reproducibility
def seed_everything(seed: int) -> None:
    """
    Sets random seeds for reproducibility across random, numpy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU


# Define the CNN model
class ConvNeuralNet(nn.Module):
    """
    A convolutional neural network for MNIST digit classification.

    Architecture:
    - Two convolutional blocks with ReLU, MaxPool and Dropout.
    - Fully connected classifier with two hidden layers and dropout.
    - Outputs class logits.
    """

    def __init__(self, dropout1=0.5, dropout2=0.25):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(p=dropout2),
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(p=dropout1),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x


# Select the best available device
def select_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Training function
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


# Validation function
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


# Main training pipeline
def run_training_pipeline(params=None):
    # Default hyperparameters
    default_params = {
        "epochs": 30,  # total training epochs
        "scheduler_step": 2,  # step interval for learning rate scheduler
        "scheduler_gamma": 0.8,  # decay factor for learning rate
        "learning_rate": 0.001,  # initial learning rate
        "weight_decay": 0.0001,  # L2 regularization strength
        "batch_size": 128,  # samples per batch
        "num_workers": 4,  # data loading workers
        "dropout1": 0.5,  # dropout rate for first dropout layer
        "dropout2": 0.25,  # dropout rate for second dropout layer
        "patience": 10,  # epochs to wait for improvement before early stopping
        "min_delta": 0.0001,  # minimum loss improvement to reset early stopping
        "min_save_accuracy": 99.0,  # minimum accuracy required to save checkpoint
    }

    # Update with custom parameters
    if params is not None:
        default_params.update(params)

    # Extract parameters
    epochs = default_params["epochs"]
    scheduler_step = default_params["scheduler_step"]
    scheduler_gamma = default_params["scheduler_gamma"]
    learning_rate = default_params["learning_rate"]
    weight_decay = default_params["weight_decay"]
    batch_size = default_params["batch_size"]
    num_workers = default_params["num_workers"]
    dropout1 = default_params["dropout1"]
    dropout2 = default_params["dropout2"]
    patience = default_params["patience"]
    min_delta = default_params["min_delta"]
    min_save_accuracy = default_params["min_save_accuracy"]

    # Set up model, optimizer, scheduler and data transforms
    device = select_device()
    print(f"Using device: {device}")

    model = ConvNeuralNet(dropout1=dropout1, dropout2=dropout2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    # Data augmentation for training
    transform_train = transforms.Compose(
        [
            transforms.RandomRotation(12),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    # Validation/test transformations
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # Load MNIST dataset
    train_dataset = MNIST(
        root="./data", train=True, download=True, transform=transform_train
    )
    val_dataset = MNIST(
        root="./data", train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    # Early stopping variables
    best_val_loss = float("inf")
    best_val_acc = 0
    best_model = None
    early_stopping_counter = 0

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate
        scheduler.step()

        print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.2f}")

        # Save model if validation accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model.state_dict()
            torch.save(best_model, "best_model.pt")
            print(f"New best model saved with accuracy: {val_acc:.2f}%")
            early_stopping_counter = 0
        elif val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # Early stopping check
        if early_stopping_counter >= patience:
            print(
                f"Early stopping triggered after {epoch} epochs! No improvement for {patience} epochs."
            )
            break

    # Always save the final model
    torch.save(model.state_dict(), "final_model.pt")

    # Save the best model (if it exists)
    if best_model is not None:
        torch.save(best_model, "best_model.pt")

    return best_val_acc


if __name__ == "__main__":
    print("=" * 50)
    print("MNIST CNN TRAINING")
    print("=" * 50)

    # Set random seed for reproducibility
    seed_everything(42)

    # Define different hyperparameter configurations to try
    param_grid = [
        {
            "epochs": 30,
            "dropout1": 0.3,  # Lower dropout rate in first layer
            "dropout2": 0.2,  # Lower dropout rate in second layer
            "batch_size": 128,
            "learning_rate": 0.001,
            "scheduler_gamma": 0.8,
            "scheduler_step": 2,
            "weight_decay": 0.0001,
        },
        {
            "epochs": 30,
            "dropout1": 0.5,  # Original dropout rate
            "dropout2": 0.4,  # Original dropout rate
            "batch_size": 128,
            "learning_rate": 0.0005,  # Lower learning rate
            "scheduler_gamma": 0.8,
            "scheduler_step": 2,
            "weight_decay": 0.0001,
        },
    ]

    # Run training with each configuration
    results = []
    for i, params in enumerate(param_grid):
        print(f"\n[RUN] Training config {i + 1}/{len(param_grid)}: {params}")
        accuracy = run_training_pipeline(params)
        results.append((i, accuracy, params))

    # Print summary of results
    print("\n" + "=" * 50)
    print("TRAINING RESULTS SUMMARY")
    print("=" * 50)
    for i, accuracy, params in sorted(results, key=lambda x: x[1], reverse=True):
        print(f"Config {i + 1}: Accuracy = {accuracy:.2f}%")
        print(f"Parameters: {params}")
        print("-" * 30)
