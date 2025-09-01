# This file trains our models A and B with L1 and L2 regularisation to be our simple and complex models

# imports

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

import torchvision.utils as vutils

import torchvision.models as models

from collections import OrderedDict

from torch.optim.lr_scheduler import ReduceLROnPlateau

# Imports for simplicity metrics
import gzip
import pickle
import copy

import torch.nn.utils.prune as prune

import os
from torchvision.models import resnet50

import pandas as pd

criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 64

full_trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size

trainset, valset = random_split(full_trainset, [train_size, val_size])

testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


# Load the first ResNet-50 model
model1 = models.resnet50(pretrained=False)

# Modify the final fully connected layer to match MNIST (10 classes)
model1.fc = nn.Linear(model1.fc.in_features, 10)

# Move the model to the appropriate device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model1.to(device)

# Load the second ResNet-50 model
model2 = models.resnet50(pretrained=False)

# Modify the final fully connected layer to match MNIST (10 classes)
model2.fc = nn.Linear(model2.fc.in_features, 10)

# Move the model to the appropriate device (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2.to(device)

# MODEL 1 (L2 Regularization) TRAINING
print("--- Training Model 1 (L2 Regularization) ---")

# Use distinct optimizer and scheduler for model1
optimizer1 = optim.SGD(model1.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001) # L2 is weight_decay
scheduler1 = ReduceLROnPlateau(optimizer1, mode='min', factor=0.1, patience=3)

# Initialize early stopping variables for model1
min_val_loss1 = float('inf')
epochs_no_improve1 = 0
early_stopping_patience = 5 # How many epochs to wait after last improvement.
num_epochs = 50

for epoch in range(num_epochs):
    model1.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer1.zero_grad()
        outputs = model1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer1.step()
        running_loss += loss.item()

    # Validation
    model1.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model1(inputs)
            val_loss += criterion(outputs, labels).item()

    avg_val_loss = val_loss / len(valloader)
    current_lr = optimizer1.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(trainloader):.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr}")

    scheduler1.step(avg_val_loss)

    # Checkpointing and Early Stopping for model1
    if avg_val_loss < min_val_loss1:
        print(f"Validation loss for model 1 decreased ({min_val_loss1:.4f} --> {avg_val_loss:.4f}). Saving model ...")
        torch.save(model1.state_dict(), 'ResNet50_L2_model.pth')
        min_val_loss1 = avg_val_loss
        epochs_no_improve1 = 0
    else:
        epochs_no_improve1 += 1

    if epochs_no_improve1 == early_stopping_patience:
        print("Early stopping triggered for model 1.")
        break
print('Finished Training Model 1.\n')



# MODEL 2 (L1 Regularization) TRAINING
print("--- Training Model 2 (L1 Regularization) ---")

# Use distinct optimizer and scheduler for model2
optimizer2 = optim.SGD(model2.parameters(), lr=0.01, momentum=0.9)
scheduler2 = ReduceLROnPlateau(optimizer2, mode='min', factor=0.1, patience=3)
l1_lambda = 0.00005 # L1 regularization strength

# early stopping variables for model2
min_val_loss2 = float('inf')
epochs_no_improve2 = 0

for epoch in range(num_epochs):
    model2.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer2.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, labels)
        
        # Add L1 regularization penalty
        l1_penalty = sum(p.abs().sum() for p in model2.parameters())
        loss = loss + l1_lambda * l1_penalty

        loss.backward()
        optimizer2.step()
        running_loss += loss.item()

    # Validation
    model2.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in valloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model2(inputs)
            val_loss += criterion(outputs, labels).item()

    avg_val_loss = val_loss / len(valloader)
    current_lr = optimizer2.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss/len(trainloader):.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr}")

    scheduler2.step(avg_val_loss)

    # Checkpointing and Early Stopping for model2 (Correctly using model2 variables)
    if avg_val_loss < min_val_loss2:
        print(f"Validation loss for model 2 decreased ({min_val_loss2:.4f} --> {avg_val_loss:.4f}). Saving model ...")
        torch.save(model2.state_dict(), 'ResNet50_L1_model.pth')
        min_val_loss2 = avg_val_loss
        epochs_no_improve2 = 0
    else:
        epochs_no_improve2 += 1

    if epochs_no_improve2 == early_stopping_patience:
        print("Early stopping triggered for model 2.")
        break
print('Finished Training Model 2.')

# Prune the final model after training
print("\n--- Pruning Model 2 ---")
model2.load_state_dict(torch.load('ResNet50_L1_model.pth')) # Load best weights before pruning
for module in model2.modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        prune.l1_unstructured(module, name='weight', amount=0.2) # Prune 20% of weights
        prune.remove(module, 'weight') # Make pruning permanent

# Save the final pruned model
torch.save(model2.state_dict(), 'ResNet50_L1_pruned_model.pth')
print("Saved pruned model 2.")


