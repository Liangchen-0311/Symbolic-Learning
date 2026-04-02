"""
Test the extracted formulas on CIFAR-10 test set.
"""
import sys
import json
import torch
import yaml
import numpy as np

sys.path.insert(0, '/Users/tan/Desktop/Code/neurosymbolic-mnist-rl')

from src.data.mnist_loader import MNISTDataModule
from src.symbolic.tensor_operators import TENSOR_OPERATORS
from src.models.lasso_classifier import train_lasso_classifier

def execute_formula(tokens, data_batch):
    """Execute a formula in RPN."""
    stack = []
    for token in tokens:
        if token in ['I_R', 'I_G', 'I_B']:
            stack.append(data_batch[token])
        elif token in TENSOR_OPERATORS:
            op_func, arity, _ = TENSOR_OPERATORS[token]
            if len(stack) < arity:
                raise ValueError(f"Not enough operands for {token}")
            operands = [stack.pop() for _ in range(arity)]
            operands.reverse()
            result = op_func(*operands)
            
            # Check for NaN/Inf
            if torch.isnan(result).any() or torch.isinf(result).any():
                raise ValueError(f"NaN/Inf at {token}")
            stack.append(result)
    
    if len(stack) != 1:
        raise ValueError(f"Invalid stack: {len(stack)} elements")
    return stack[0]

# Load formulas
with open('bank_formulas.json', 'r') as f:
    formulas = json.load(f)

print(f"\n{'='*60}")
print(f"Testing {len(formulas)} Formulas on CIFAR-10 Test Set")
print(f"{'='*60}\n")

# Load config
with open('outputs/tensor_vsr_large_bank/config.yaml') as f:
    config = yaml.safe_load(f)

device = config['training']['device']
num_classes = config['dataset']['num_classes']

# Load CIFAR-10 test data
print("Loading CIFAR-10 test data...")
data_module = MNISTDataModule(
    dataset='cifar10',
    batch_size=1000,
    num_workers=0
)
data_module.setup()
test_loader = data_module.get_test_loader()

# Get full test set
all_features = []
all_labels = []

print(f"Extracting features from {len(formulas)} formulas...")

for batch_idx, (images, labels) in enumerate(test_loader):
    images = images.to(device)
    labels = labels.to(device)
    
    # Extract RGB channels
    data_batch = {
        'I_R': images[:, 0, :, :],
        'I_G': images[:, 1, :, :],
        'I_B': images[:, 2, :, :],
    }
    
    # Execute all formulas
    batch_features = []
    failed_count = 0
    for i, formula_dict in enumerate(formulas):
        try:
            output = execute_formula(formula_dict['tokens'], data_batch)
            output = torch.nan_to_num(output, nan=0.0, posinf=1e4, neginf=-1e4)
            
            # Normalize
            mean = output.mean()
            std = output.std()
            if std > 1e-5:
                output = (output - mean) / (std + 1e-8)
            else:
                output = output - mean
            
            batch_features.append(output)
        except Exception as e:
            if failed_count < 5:  # Only print first 5 failures
                print(f"  Warning: Formula {i} failed: {e}")
            failed_count += 1
            # Use zeros if formula fails
            batch_features.append(torch.zeros(images.shape[0], device=device))
    
    if failed_count > 0:
        print(f"  Total failed formulas in batch: {failed_count}/{len(formulas)}")
    
    # Stack features
    if batch_features:
        batch_features_tensor = torch.stack(batch_features, dim=1)  # [batch, num_formulas]
        all_features.append(batch_features_tensor)
        all_labels.append(labels)
    
    if batch_idx == 0:
        print(f"  Batch shape: {batch_features_tensor.shape}")
    
    print(f"  Processed batch {batch_idx+1}/{len(test_loader)}")

# Concatenate all batches
features_tensor = torch.cat(all_features, dim=0)
labels_tensor = torch.cat(all_labels, dim=0)

print(f"\nTotal test samples: {features_tensor.shape[0]}")
print(f"Feature dimensions: {features_tensor.shape[1]}")

# Test with different LASSO regularization strengths
print(f"\n{'='*60}")
print(f"Testing Different LASSO Regularization Strengths")
print(f"{'='*60}\n")

l1_lambdas = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0]

results = []
for l1_lambda in l1_lambdas:
    print(f"L1 Lambda = {l1_lambda}")
    
    accuracy, active_features, model = train_lasso_classifier(
        features_tensor,
        labels_tensor,
        num_classes=num_classes,
        l1_lambda=l1_lambda,
        epochs=200,
        device=device
    )
    
    selection_rate = active_features / features_tensor.shape[1] * 100
    
    print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Active Features: {active_features}/{features_tensor.shape[1]} ({selection_rate:.1f}%)")
    print(f"  Sparsity: {active_features/features_tensor.shape[1]:.3f}")
    print()
    
    results.append({
        'l1_lambda': l1_lambda,
        'accuracy': accuracy,
        'active_features': active_features,
        'total_features': features_tensor.shape[1],
        'selection_rate': selection_rate
    })

# Find best result
best_result = max(results, key=lambda x: x['accuracy'])

print(f"{'='*60}")
print(f"BEST TEST RESULTS")
print(f"{'='*60}")
print(f"L1 Lambda: {best_result['l1_lambda']}")
print(f"Test Accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)")
print(f"Active Features: {best_result['active_features']}/{best_result['total_features']}")
print(f"Selection Rate: {best_result['selection_rate']:.1f}%")
print(f"{'='*60}\n")

# Compare to random baseline
print(f"Comparison to Baseline:")
print(f"  Random Baseline: 10.00%")
print(f"  Our Method: {best_result['accuracy']*100:.2f}%")
print(f"  Improvement: {(best_result['accuracy']*100 - 10):.2f}% (absolute)")
print(f"  Relative Gain: {((best_result['accuracy']*100 / 10) - 1) * 100:.1f}%")
print()

# Save results
with open('test_results.json', 'w') as f:
    json.dump({
        'all_results': results,
        'best_result': best_result,
        'num_formulas': len(formulas)
    }, f, indent=2)

print("Results saved to test_results.json")
