#!/usr/bin/env python3
"""
Automatically generate revised configs based on feedback analysis.

This script:
1. Reads the best performing configs from analysis
2. Generates new configs to explore around the best settings
3. Creates a new iteration batch
"""

import argparse
import yaml
import json
from pathlib import Path
import shutil


def load_analysis(analysis_file):
    """Load feedback analysis results."""
    with open(analysis_file, 'r') as f:
        return json.load(f)


def generate_revised_configs(analysis, output_dir):
    """Generate revised configs based on best performers."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    revisions = []
    
    # CIFAR-10 revisions
    best_c10 = analysis.get('best_cifar10')
    if best_c10:
        print(f"\nBest CIFAR-10: {best_c10['name']} with {best_c10['accuracy']:.2f}%")
        
        # Load base config
        base_config_path = Path('configs/tensor_vsr_m1_cifar10.yaml')
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)
        
        # Get best L1 lambda
        best_l1 = best_c10['params'].get('l1_lambda', 0.01)
        print(f"  Best L1: {best_l1}")
        
        # Generate variations around best L1
        if best_l1 == 0.001:
            # Try even lower and slightly higher
            new_l1_values = [0.0005, 0.0015, 0.002]
        elif best_l1 == 0.01:
            # Try around baseline
            new_l1_values = [0.005, 0.015, 0.02]
        elif best_l1 == 0.1:
            # Try lower from high
            new_l1_values = [0.05, 0.075, 0.15]
        else:
            new_l1_values = [best_l1 * 0.5, best_l1 * 1.5, best_l1 * 2.0]
        
        for l1_val in new_l1_values:
            new_config = base_config.copy()
            new_config['training']['l1_lambda'] = l1_val
            
            # Save config
            config_name = f"tensor_vsr_m1_cifar10_l1_{l1_val:.4f}.yaml"
            config_path = output_path / config_name
            
            with open(config_path, 'w') as f:
                yaml.dump(new_config, f, default_flow_style=False)
            
            revisions.append({
                'dataset': 'cifar10',
                'config': str(config_path),
                'name': f'l1_{l1_val:.4f}',
                'l1_lambda': l1_val
            })
            
            print(f"  Created: {config_name}")
        
        # Try larger feature bank if utilization was high
        if best_c10['utilization'] > 0.75:
            new_bank_sizes = [15, 20, 25]
            for bank_size in new_bank_sizes:
                new_config = base_config.copy()
                new_config['training']['feature_bank_size'] = bank_size
                new_config['training']['l1_lambda'] = best_l1  # Use best L1
                
                config_name = f"tensor_vsr_m1_cifar10_bank{bank_size}.yaml"
                config_path = output_path / config_name
                
                with open(config_path, 'w') as f:
                    yaml.dump(new_config, f, default_flow_style=False)
                
                revisions.append({
                    'dataset': 'cifar10',
                    'config': str(config_path),
                    'name': f'bank{bank_size}',
                    'feature_bank_size': bank_size
                })
                
                print(f"  Created: {config_name}")
    
    # CIFAR-100 revisions
    best_c100 = analysis.get('best_cifar100')
    if best_c100:
        print(f"\nBest CIFAR-100: {best_c100['name']} with {best_c100['accuracy']:.2f}%")
        
        # Load base config
        base_config_path = Path('configs/tensor_vsr_m1_cifar100.yaml')
        with open(base_config_path) as f:
            base_config = yaml.safe_load(f)
        
        # If accuracy is low, try lower thresholds
        if best_c100['accuracy'] < 8:
            new_thresholds = [0.015, 0.01, 0.005]
            for threshold in new_thresholds:
                new_config = base_config.copy()
                new_config['training']['min_accuracy'] = threshold
                
                config_name = f"tensor_vsr_m1_cifar100_acc_{threshold:.3f}.yaml"
                config_path = output_path / config_name
                
                with open(config_path, 'w') as f:
                    yaml.dump(new_config, f, default_flow_style=False)
                
                revisions.append({
                    'dataset': 'cifar100',
                    'config': str(config_path),
                    'name': f'acc_{threshold:.3f}',
                    'min_accuracy': threshold
                })
                
                print(f"  Created: {config_name}")
        
        # Try larger feature banks
        if best_c100['utilization'] > 0.7:
            new_bank_sizes = [20, 25, 30]
            for bank_size in new_bank_sizes:
                new_config = base_config.copy()
                new_config['training']['feature_bank_size'] = bank_size
                
                config_name = f"tensor_vsr_m1_cifar100_bank{bank_size}.yaml"
                config_path = output_path / config_name
                
                with open(config_path, 'w') as f:
                    yaml.dump(new_config, f, default_flow_style=False)
                
                revisions.append({
                    'dataset': 'cifar100',
                    'config': str(config_path),
                    'name': f'bank{bank_size}',
                    'feature_bank_size': bank_size
                })
                
                print(f"  Created: {config_name}")
    
    # Save revision plan
    revision_plan_path = output_path / 'revision_plan.json'
    with open(revision_plan_path, 'w') as f:
        json.dump(revisions, f, indent=2)
    
    print(f"\nRevision plan saved to: {revision_plan_path}")
    print(f"Total revised configs: {len(revisions)}")
    
    return revisions


def generate_iteration_script(revisions, output_dir):
    """Generate a new iteration script for revised configs."""
    script_path = Path(output_dir) / 'iterate_revised.sh'
    
    with open(script_path, 'w') as f:
        f.write('#!/bin/bash\n\n')
        f.write('# Auto-generated revision iteration script\n\n')
        f.write('TIMESTAMP=$(date +%Y%m%d_%H%M%S)\n')
        f.write('OUTPUT_DIR="outputs/m1_iterations_revised/$TIMESTAMP"\n')
        f.write('mkdir -p $OUTPUT_DIR\n\n')
        
        for i, rev in enumerate(revisions):
            f.write(f'# Revision {i+1}: {rev["dataset"]} - {rev["name"]}\n')
            f.write(f'python3 experiments/train_tensor_vsr.py \\\n')
            f.write(f'    --config {rev["config"]} \\\n')
            f.write(f'    --dataset {rev["dataset"]} \\\n')
            f.write(f'    --device mps \\\n')
            f.write(f'    --output_dir $OUTPUT_DIR/{rev["dataset"]}_{rev["name"]}\n\n')
            
            f.write(f'python3 experiments/quick_eval.py \\\n')
            f.write(f'    --feature_bank $OUTPUT_DIR/{rev["dataset"]}_{rev["name"]}/best_model.pth \\\n')
            f.write(f'    --config {rev["config"]} \\\n')
            f.write(f'    --dataset {rev["dataset"]}\n\n')
    
    # Make executable
    script_path.chmod(0o755)
    print(f"\nIteration script saved to: {script_path}")
    print(f"Run with: {script_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto-generate revised configs from feedback')
    parser.add_argument('--analysis', type=str, required=True,
                       help='Path to feedback_analysis.json')
    parser.add_argument('--output_dir', type=str, default='configs/revised',
                       help='Directory to save revised configs')
    parser.add_argument('--generate_script', action='store_true',
                       help='Generate iteration script for revised configs')
    args = parser.parse_args()
    
    # Load analysis
    analysis = load_analysis(args.analysis)
    
    # Generate revised configs
    revisions = generate_revised_configs(analysis, args.output_dir)
    
    # Generate iteration script if requested
    if args.generate_script and revisions:
        generate_iteration_script(revisions, args.output_dir)
    
    print("\n" + "=" * 60)
    print("REVISION GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Review configs in: {args.output_dir}")
    if args.generate_script:
        print(f"  2. Run: {args.output_dir}/iterate_revised.sh")
    print(f"  3. Analyze new results with analyze_feedback.py")
    print("=" * 60)
