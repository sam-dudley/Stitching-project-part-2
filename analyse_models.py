
# All the necessary imports

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

batch_size = 16

full_trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)

train_size = int(0.8 * len(full_trainset))
val_size = len(full_trainset) - train_size

trainset, valset = random_split(full_trainset, [train_size, val_size])

testset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)


# All the simplicity metrics

def compute_gradient_norm(model, loss_fn, testloader):
    #Computes the average gradient norm over the test set.
    # Ensure model is in training mode for gradient calculation,
    # or handle dropout/batchnorm if you truly want eval behavior but with gradients.
    # For a true gradient norm related to generalization, model.eval() is fine for layer behavior,
    # but gradients are still computed.
    model.eval() # Keep eval mode for consistent layer behavior during inference

    total_norm = 0.0
    num_batches = 0
    device = next(model.parameters()).device # Get device once

    for data, target in testloader:
        data, target = data.to(device), target.to(device)

        model.zero_grad() # Clear previous gradients
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward() # Compute gradients

        # Calculate the L2 norm of the gradients
        # Use torch.norm(p.grad) for L2 norm of individual gradient tensors
        # Or torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).norm()
        # The sum of squared individual norms is also correct for the overall L2 norm.
        batch_norm = torch.sqrt(sum(p.grad.norm()**2 for p in model.parameters() if p.grad is not None))
        total_norm += batch_norm.item()
        num_batches += 1

    return total_norm / num_batches if num_batches > 0 else 0.0


def compute_loss_sharpness(model, loss_fn, testloader, epsilon=1e-3, num_runs=5):
    
    # Computes loss sharpness by averaging over multiple runs to get a more
    # stable, less noisy result.
   
    run_sharpness_values = []

    for _ in range(num_runs):
        model.eval()
        total_sharpness_for_run = 0.0
        num_batches = 0
        device = next(model.parameters()).device
        original_params = [p.clone().detach() for p in model.parameters()]

        for data, target in testloader:
            data, target = data.to(device), target.to(device)

            with torch.no_grad():
                output = model(data)
                original_loss = loss_fn(output, target).item()

                # Perturb parameters
                for i, p in enumerate(model.parameters()):
                    p.add_(epsilon * torch.randn_like(p))

                # Compute perturbed loss
                perturbed_output = model(data)
                perturbed_loss = loss_fn(perturbed_output, target).item()

                # Restore original parameters
                for i, p in enumerate(model.parameters()):
                    p.copy_(original_params[i])
            
            total_sharpness_for_run += perturbed_loss - original_loss
            num_batches += 1

        run_sharpness_values.append(total_sharpness_for_run / num_batches)

    # Return the average of all the runs
    return sum(run_sharpness_values) / len(run_sharpness_values)

def compress_model_numpy(model):
    """Compresses the models parameters into a bit stream."""
    with torch.no_grad():
        weights = np.concatenate([p.cpu().detach().numpy().flatten() for p in model.parameters()])
    model_bytes = pickle.dumps(weights)
    compressed = gzip.compress(model_bytes)
    compressed_bits = len(compressed) * 8  # Size in bits
    uncompressed_bits = len(model_bytes) * 8
    compression_ratio = uncompressed_bits / compressed_bits if compressed_bits > 0 else float('inf')
    return compressed_bits, compression_ratio

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_nonzero_parameters(model, tolerance=1e-9):
    # Counts parameters that are not close to zero
    return sum((p.abs() > tolerance).sum().item() for p in model.parameters() if p.requires_grad)

def get_hessian_vector_product(model, loss, vector):
    """
    Computes the Hessian-vector product (Hv) for a given loss and model.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True)
    
    # Calculate the dot product of the gradients and the vector
    grad_vector_dot_product = sum(g.mul(v).sum() for g, v in zip(grads, vector))
    
    # Compute the gradient of the dot product with respect to the parameters, which is the HVP
    hvp = torch.autograd.grad(grad_vector_dot_product, params, retain_graph=True)
    
    return hvp

def compute_hessian_top_eigenvalue(model, loss_fn, dataloader, num_iterations=20):
    """
    Computes the top eigenvalue of the Hessian using the Power Iteration method.
    """
    print(f"  Calculating Hessian top eigenvalue ({num_iterations} iterations)...")
    device = next(model.parameters()).device
    
    # Initialize a random vector with the same dimensions as the model parameters
    vector = [torch.randn_like(p, device=device) for p in model.parameters() if p.requires_grad]
    
    # Power Iteration loop
    for _ in range(num_iterations):
        model.zero_grad()
        
        # We need a loss value calculated on a batch of data
        try:
            data, target = next(iter(dataloader))
            data, target = data.to(device), target.to(device)
            loss = loss_fn(model(data), target)
        except StopIteration:
            print("Warning: Dataloader exhausted during power iteration. Re-initializing.")
            dataloader_iter = iter(dataloader)
            data, target = next(dataloader_iter)
            data, target = data.to(device), target.to(device)
            loss = loss_fn(model(data), target)
        
        hvp = get_hessian_vector_product(model, loss, vector)
        
        # Calculate the eigenvalue (Rayleigh quotient)
        eigenvalue = sum(v.mul(h_v).sum() for v, h_v in zip(vector, hvp)).item()
        
        # Normalize the vector for the next iteration
        norm = torch.sqrt(sum(h_v.pow(2).sum() for h_v in hvp))
        vector = [h_v / (norm + 1e-9) for h_v in hvp]
        
    return eigenvalue

def compute_hessian_trace(model, loss_fn, dataloader, num_samples=10):
    """
    Computes the trace of the Hessian using Hutchinson's method.
    """
    print(f"  Calculating Hessian trace (averaging over {num_samples} samples)...")
    device = next(model.parameters()).device
    trace_estimates = []

    for _ in range(num_samples):
        model.zero_grad()

        # Generate a random Rademacher vector (elements are -1 or 1)
        rademacher_vector = [torch.randint_like(p, high=2, device=device) * 2 - 1 for p in model.parameters() if p.requires_grad]

        # We need a loss value calculated on a batch of data
        try:
            data, target = next(iter(dataloader))
            data, target = data.to(device), target.to(device)
            loss = loss_fn(model(data), target)
        except StopIteration:
            print("Warning: Dataloader exhausted during trace estimation. Re-initializing.")
            dataloader_iter = iter(dataloader)
            data, target = next(dataloader_iter)
            data, target = data.to(device), target.to(device)
            loss = loss_fn(model(data), target)

        hvp = get_hessian_vector_product(model, loss, rademacher_vector)
        
        # Calculate z^T * H * z
        trace_estimate = sum(v.mul(h_v).sum() for v, h_v in zip(rademacher_vector, hvp)).item()
        trace_estimates.append(trace_estimate)

    return sum(trace_estimates) / len(trace_estimates)
    
    
    # All the different classes of models
    
    #stitching layer
class StitchingLayer(nn.Module):
    """
    A low-capacity stitching layer that uses a bottleneck to reduce
    the number of parameters.
    """
    def __init__(self, in_channels, out_channels, bottleneck_size=64):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, bottleneck_size, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bottleneck_size, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(x)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
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
        
def calculate_all_metrics(model, test_loader, loss_fn):
    """
    Calculates all performance and simplicity metrics for a given model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    metrics = {}
    
    # --- Performance Metric: Test Accuracy ---
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    metrics['accuracy'] = 100 * correct / total
    
    # --- Simplicity Metrics (using your defined functions) ---
    metrics['grad_norm'] = compute_gradient_norm(model, loss_fn, test_loader)
    metrics['loss_sharpness'] = compute_loss_sharpness(model, loss_fn, test_loader)
    
    compressed_bits, _ = compress_model_numpy(model)
    metrics['compressed_size_bits'] = compressed_bits
    
    metrics['total_params'] = count_parameters(model)
    metrics['nonzero_params'] = count_nonzero_parameters(model)
    
    metrics['hessian_top_eigenvalue'] = compute_hessian_top_eigenvalue(model, loss_fn, test_loader)
    metrics['hessian_trace'] = compute_hessian_trace(model, loss_fn, test_loader)
    
    return metrics

def run_full_analysis(model_a_path, model_b_path, stitched_models_dir, test_loader):
    """
    Loads all trained models, calculates metrics for each, and returns a DataFrame.
    """
    print("🔬 Starting final analysis of all models...")
    
    all_results = []
    loss_fn = nn.CrossEntropyLoss()

    # --- Analyze Model A (Simpler) ---
    print("Analyzing Model A (L1 Pruned)...")
    model_a = resnet50(num_classes=10)
    model_a.load_state_dict(torch.load(model_a_path))
    model_a_metrics = calculate_all_metrics(model_a, test_loader, loss_fn)
    model_a_metrics['model'] = 'Model A (Simple)'
    model_a_metrics['stitch_location'] = 0 # Use 0 for the base model A
    all_results.append(model_a_metrics)
    
    # --- Analyze Model B (Complex) ---
    print("Analyzing Model B (L2)...")
    model_b = resnet50(num_classes=10)
    model_b.load_state_dict(torch.load(model_b_path))
    model_b_metrics = calculate_all_metrics(model_b, test_loader, loss_fn)
    model_b_metrics['model'] = 'Model B (Complex)'
    model_b_metrics['stitch_location'] = 17 # Use 17 for the base model B
    all_results.append(model_b_metrics)
    
    # --- Analyze all 16 Stitched Models ---
    for i in range(1, 17):
        stitch_location = i
        print(f"Analyzing Stitched Model at location {stitch_location}...")
        
        # Instantiate the correct stitched model architecture
        model_stitched = StitchedResNet50(stitch_location=stitch_location, num_classes=10)
        
        # Load its trained weights
        model_path = os.path.join(stitched_models_dir, f'stitched_resnet50_loc_{stitch_location}.pth')
        model_stitched.load_state_dict(torch.load(model_path))
        
        # Calculate metrics
        stitched_metrics = calculate_all_metrics(model_stitched, test_loader, loss_fn)
        stitched_metrics['model'] = f'Stitched_{stitch_location}'
        stitched_metrics['stitch_location'] = stitch_location
        all_results.append(stitched_metrics)
        
        # Explicitly delete the model and clear the CUDA cache
        del model_stitched
        torch.cuda.empty_cache()
        
    print("\n✅ Analysis complete.")
    
    # Convert results to a pandas DataFrame for easy analysis and plotting
    results_df = pd.DataFrame(all_results)
    return results_df

# Now upload the already trained and stitched models and carry out the analysis

if __name__ == '__main__':
    MODEL_A_PATH = 'ResNet50_L1_pruned_model.pth'
    MODEL_B_PATH = 'ResNet50_L2_model.pth'
    STITCHED_MODELS_DIR = 'resnet50_stitched_models'

    # Run the analysis
    final_results_df = run_full_analysis(
        model_a_path=MODEL_A_PATH,
        model_b_path=MODEL_B_PATH,
        stitched_models_dir=STITCHED_MODELS_DIR,
        test_loader=testloader
    )

    # Save the final results
    final_results_df.to_csv('final_experiment_metrics.csv', index=False)
    print("\nFinal results DataFrame created and saved to 'final_experiment_metrics.csv'")
    print(final_results_df)
        
  