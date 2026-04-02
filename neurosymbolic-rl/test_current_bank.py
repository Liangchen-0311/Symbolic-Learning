"""
Test evaluation of current feature bank with LASSO selection.
"""
import sys
import torch
import yaml
from pathlib import Path

# Add project to path
sys.path.insert(0, '/Users/tan/Desktop/Code/neurosymbolic-mnist-rl')

from src.data.mnist_loader import MNISTDataModule
from src.symbolic.tensor_evaluator import TensorProgramEvaluator
from src.rl.tensor_environment_large_bank import TensorTokenVocabulary

def test_feature_bank(checkpoint_path, config_path):
    """Test current feature bank with LASSO."""
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    device = config['training']['device']
    print(f"\n{'='*60}")
    print(f"Testing Feature Bank from Checkpoint")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}\n")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get feature bank
    feature_bank = checkpoint.get('feature_bank', [])
    print(f"Feature Bank Size: {len(feature_bank)}")
    
    if len(feature_bank) == 0:
        print("No formulas in feature bank yet!")
        return
    
    # Show some formulas
    print(f"\n{'='*60}")
    print(f"Sample Formulas (Top 10 by accuracy):")
    print(f"{'='*60}")
    sorted_formulas = sorted(feature_bank, key=lambda x: x.get('accuracy', 0), reverse=True)
    for i, formula in enumerate(sorted_formulas[:10]):
        acc = formula.get('accuracy', 0)
        formula_str = formula.get('formula_str', 'N/A')
        length = formula.get('length', 0)
        print(f"{i+1:2d}. [{acc:.3f}] (len={length:2d}) {formula_str}")
    
    # Load data
    print(f"\n{'='*60}")
    print(f"Loading CIFAR-10 test data...")
    print(f"{'='*60}")
    
    data_module = MNISTDataModule(
        dataset_name=config['dataset']['name'],
        batch_size=512,
        data_dir='./data',
        num_workers=0
    )
    data_module.setup()
    test_loader = data_module.test_dataloader()
    
    # Get test batch
    images, labels = next(iter(test_loader))
    images = images.to(device)
    labels = labels.to(device)
    
    # Extract RGB channels
    data_batch = {
        'I_R': images[:, 0, :, :],
        'I_G': images[:, 1, :, :],
        'I_B': images[:, 2, :, :],
    }
    
    print(f"Test batch size: {images.shape[0]}")
    
    # Evaluate with LASSO
    print(f"\n{'='*60}")
    print(f"Running LASSO Evaluation...")
    print(f"{'='*60}")
    
    vocabulary = TensorTokenVocabulary()
    evaluator = TensorProgramEvaluator(
        num_classes=config['dataset']['num_classes'],
        device=device
    )
    
    # Test different L1 lambdas
    l1_lambdas = [0.01, 0.1, 0.5, 1.0]
    
    for l1_lambda in l1_lambdas:
        accuracy, active_features, metrics = evaluator.evaluate_feature_bank(
            feature_bank=feature_bank,
            vocabulary=vocabulary,
            data_batch=data_batch,
            labels=labels,
            l1_lambda=l1_lambda,
            lasso_epochs=100
        )
        
        selection_rate = active_features / len(feature_bank) * 100
        
        print(f"\nL1 Lambda: {l1_lambda}")
        print(f"  Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
        print(f"  Active Features: {active_features}/{len(feature_bank)} ({selection_rate:.1f}%)")
        print(f"  Sparsity: {metrics.get('sparsity', 0):.3f}")
    
    print(f"\n{'='*60}")
    print(f"Evaluation Complete!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    checkpoint_path = 'outputs/tensor_vsr_large_bank/checkpoint_iter_50.pt'
    config_path = 'outputs/tensor_vsr_large_bank/config.yaml'
    
    test_feature_bank(checkpoint_path, config_path)
