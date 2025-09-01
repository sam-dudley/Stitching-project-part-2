

# Imports

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

# Data imports

criterion = nn.CrossEntropyLoss()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

batch_size = 16

full_trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size

trainset, valset = random_split(full_trainset, [train_size, val_size])

testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


# Defining the classes needed for stitching

# Stitching layer
class StitchingLayer(nn.Module):
    """
    Stitching layer as described in the paper, with two BatchNorm layers
    for normalization.
    """
    def __init__(self, in_channels, out_channels):
        super(StitchingLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1x1(x)
        x = self.bn2(x)
        return x

class Bottleneck(nn.Module):
    """
    The Bottleneck block used in ResNet-50.
    """
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out += identity
        out = self.relu(out)

        return out

class StitchedResNet50(nn.Module):
    def __init__(self, stitch_location, num_classes=10):
        super(StitchedResNet50, self).__init__()

        if not 1 <= stitch_location <= 16:
            raise ValueError("Stitch location must be an integer between 1 and 16.")
        self.stitch_location = stitch_location
        
        block_counts = [3, 4, 6, 3]
        self.in_channels = 64

        # Initial layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Sequentially define all 16 bottleneck blocks
        self.blocks = nn.ModuleList()
        self.block_layer_map = [] # To map flat index back to layer.block name
        
        ch = 64
        current_block_count = 0
        for i, num_blocks in enumerate(block_counts):
            for j in range(num_blocks):
                stride = 2 if j == 0 and i > 0 else 1
                downsample = None
                if stride != 1 or self.in_channels != ch * Bottleneck.expansion:
                    downsample = nn.Sequential(
                        nn.Conv2d(self.in_channels, ch * Bottleneck.expansion, kernel_size=1, stride=stride, bias=False),
                        nn.BatchNorm2d(ch * Bottleneck.expansion),
                    )
                
                self.blocks.append(Bottleneck(self.in_channels, ch, stride, downsample))
                self.in_channels = ch * Bottleneck.expansion
                self.block_layer_map.append(f'layer{i+1}.{j}')
          
            ch *= 2

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)
        
        # Identify the block right before the stitch
        block_before_stitch = self.blocks[self.stitch_location - 1]
        stitch_in_ch = block_before_stitch.conv3.out_channels
        
        # Handle the edge case for the last block
        if self.stitch_location < 16:
            block_after_stitch = self.blocks[self.stitch_location]
            stitch_out_ch = block_after_stitch.conv1.in_channels
        else:  # This is the final stitch location (16)
            # The next layer is the avgpool, which doesn't change the number of channels.
            # So the stitch layer's input and output dimensions are the same.
            stitch_out_ch = stitch_in_ch

        # The input to the stitch layer is the output of the block before it
        stitch_in_ch = block_before_stitch.conv3.out_channels

        # Now, create the stitching layer with the correct dimensions
        self.stitching_layer = StitchingLayer(stitch_in_ch, stitch_out_ch)
    
    def _get_input_channels_for_block(self, block_index):
        # Helper to determine the expected input channels for any block in the sequence
        if block_index > 16: # after the last block
             return 512 * Bottleneck.expansion
        
        # This logic determines the `in_channels` for any given block based on ResNet50's structure
        if block_index <= 3: # layer1
            return 256 if block_index > 1 else 64
        elif block_index <= 7: # layer2
            return 512 if block_index > 4 else 256
        elif block_index <= 13: # layer3
            return 1024 if block_index > 8 else 512
        else: # layer4
            return 2048 if block_index > 14 else 1024


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        for i, block in enumerate(self.blocks):
            x = block(x)
            if (i + 1) == self.stitch_location:
                x = self.stitching_layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def stitch_models_resnet50(model_a, model_b, stitch_location, num_classes=10):
    """
    Creates and populates a stitched ResNet-50 model.
    """
    # 1. Create the stitched model architecture
    stitched_model = StitchedResNet50(stitch_location=stitch_location, num_classes=num_classes)
    
    # 2. Get state dictionaries
    sd_a = model_a.state_dict()
    sd_b = model_b.state_dict()
    sd_stitched = stitched_model.state_dict()

    # 3. Copy initial conv layers from model_a
    for key in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var']:
        sd_stitched[key] = sd_a[key]
        
    # 4. Copy blocks from model_a and model_b based on stitch_location
    for i in range(16):
        block_name = stitched_model.block_layer_map[i] # e.g., 'layer2.3'
        
        # Determine source model based on position relative to stitch_location
        source_sd = sd_a if (i + 1) <= stitch_location else sd_b
        
        # Find all keys for the current block and copy them
        for key in source_sd:
            if key.startswith(block_name):
                # The key in our model is `blocks.i.param`
                stitched_key = key.replace(block_name, f'blocks.{i}')
                if stitched_key in sd_stitched:
                    sd_stitched[stitched_key] = source_sd[key]

    # 5. Copy the final fully connected layer from model_b
    if 'fc.weight' in sd_b:
        sd_stitched['fc.weight'] = sd_b['fc.weight']
        sd_stitched['fc.bias'] = sd_b['fc.bias']

    # 6. Load the new state_dict
    stitched_model.load_state_dict(sd_stitched)
    
    return stitched_model


def train_stitching_layer(model, train_loader, val_loader, epochs=10, initial_lr=1e-3):
    # Use the GPU if available, which is necessary for running on Hydra's GPU nodes
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    stitching_params = [p for name, p in model.named_parameters() if 'stitching_layer' in name]
    
    optimizer = optim.SGD(stitching_params, lr=initial_lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    
    # Reduces the learning rate when validation loss stops improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=2)

    print("--- Starting Training of Stitching Layer ---")
    
    best_val_loss = float('inf')
    best_model_weights = None

    # Training
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {running_loss / len(train_loader):.4f}, Validation Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            print(f"New best model saved with validation loss: {best_val_loss:.4f}")

    print("--- Finished Training ---")
    
    # Load the best model weights back into the model
    model.load_state_dict(best_model_weights)
    return model
    
    
# Automates stitching, training, and saving for every ResNet-50 location.
# Model_a is simple, model_b is complex
def run_stitching_experiment(model_a, model_b, train_loader, val_loader, output_dir='trained_models'):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a.to(device)
    model_b.to(device)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    num_stitch_locations = 16 
    print(f"Starting full stitching experiment for {num_stitch_locations} locations...")
    print("="*60)

    for i in range(1, num_stitch_locations + 1):
        stitch_location = i
        print(f"Processing Stitch Location: {stitch_location}/{num_stitch_locations}")

        print(f"Step 1: Creating stitched model at location {stitch_location}...")
        stitched_model = stitch_models_resnet50(model_a, model_b, stitch_location=stitch_location)
        
        print(f"Step 2: Training the stitching layer...")
        trained_model = train_stitching_layer(
            model=stitched_model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=15, 
            initial_lr=1e-3
        )
        
        model_save_path = os.path.join(output_dir, f'stitched_resnet50_loc_{stitch_location}.pth')
        torch.save(trained_model.state_dict(), model_save_path)
        print(f"   Step 3: Trained model saved to {model_save_path}")
        print("-"*60)
        
    print("Full stitching experiment complete. All models trained and saved.")


# Main execution block to run the experiment after training Model 1 and Model 2

if __name__ == '__main__':
	# Load the best versions of your trained models
	print("--- Loading final trained models for stitching experiment ---")

	# Model A (Simpler Model): L1 Regularization + Pruning
	model_A_simple = resnet50(num_classes=10)
	model_A_simple.load_state_dict(torch.load('ResNet50_L1_pruned_model.pth'))
	print("Loaded simpler model (L1 + Pruned) from 'ResNet50_L1_pruned_model.pth'")

	# Model B (Complex Model): L2 Regularization
	model_B_complex = resnet50(num_classes=10)
	model_B_complex.load_state_dict(torch.load('ResNet50_L2_model.pth'))
	print("Loaded complex model (L2) from 'ResNet50_L2_model.pth'")

	# Run the entire experiment
	run_stitching_experiment(
		model_a=model_A_simple,
    		model_b=model_B_complex,
   	 	train_loader=trainloader,
   		val_loader=valloader,
   		output_dir='resnet50_stitched_models'
	 )
    
	print("--- All stitched models trained and saved. ---")


