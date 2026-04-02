"""
Comprehensive evaluation metrics for VSR experiments.

This module provides metrics to compare:
- MLP (black-box baseline)
- VSR+RL (symbolic regression with RL)
- VSR+RL+LASSO (ensemble method)

Key metrics:
1. Performance: R², RMSE, MAE
2. Efficiency: Parameters, speed, memory
3. Interpretability: Formula complexity, human readability
"""

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from typing import Dict, Optional
import time


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    n_params: int,
    n_terms: Optional[int] = None,
    inference_time: Optional[float] = None
) -> Dict:
    """
    Compute comprehensive metrics for a regression model.
    
    Args:
        y_true: Ground truth values [n_samples]
        y_pred: Predicted values [n_samples]
        model_name: Name of the method (e.g., "MLP", "VSR+RL")
        n_params: Number of trainable parameters
        n_terms: Number of terms in formula (None for MLP)
        inference_time: Time to make predictions (seconds)
    
    Returns:
        Dictionary with all metrics
    """
    # 1. Performance Metrics
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Normalized metrics (for cross-dataset comparison)
    y_range = y_true.max() - y_true.min()
    if y_range > 0:
        nmse = mse / (y_range ** 2)
        nmae = mae / y_range
    else:
        nmse = mse
        nmae = mae
    
    # 2. Efficiency Metrics
    # Parameter efficiency: R² per log(parameter)
    # Uses log to penalize parameter count reasonably
    param_efficiency = r2 / np.log10(n_params + 1)
    
    # 3. Interpretability Score
    if n_terms is not None:
        # Formula complexity penalty
        complexity_penalty = np.log10(n_terms + 1)
        base_score = 1.0 / (1.0 + complexity_penalty)
        
        # Bonus for simplicity
        if n_terms <= 5:
            bonus = 0.3  # Very simple
        elif n_terms <= 10:
            bonus = 0.2  # Medium
        elif n_terms <= 20:
            bonus = 0.1  # Complex
        else:
            bonus = 0.0  # Too complex
        
        interpretability = min(1.0, base_score + bonus)
    else:
        # MLP or other black-box method
        interpretability = 0.0
    
    # 4. Composite Score
    # Weights: 40% performance, 30% efficiency, 30% interpretability
    composite_score = (
        0.4 * r2 +
        0.3 * min(1.0, param_efficiency) +
        0.3 * interpretability
    ) * 100  # Scale to 0-100
    
    return {
        'model': model_name,
        'n_params': int(n_params),
        'n_terms': int(n_terms) if n_terms is not None else None,
        'r2': float(r2),
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'nmse': float(nmse),
        'nmae': float(nmae),
        'param_efficiency': float(param_efficiency),
        'interpretability': float(interpretability),
        'composite_score': float(composite_score),
        'inference_time': float(inference_time) if inference_time else None
    }


def compare_methods(results_dict: Dict[str, Dict], dataset_name: str = "Dataset"):
    """
    Print comprehensive comparison table for multiple methods.
    
    Args:
        results_dict: {method_name: metrics_dict}
        dataset_name: Name of the dataset
    """
    print("\n" + "="*90)
    print(f"COMPREHENSIVE COMPARISON: {dataset_name.upper()}")
    print("="*90)
    
    # Get MLP baseline for competitiveness check
    mlp_r2 = results_dict.get('MLP', {}).get('r2', 0)
    mlp_eff = results_dict.get('MLP', {}).get('param_efficiency', 1)
    
    # 1. Performance Metrics
    print("\n📊 1. PERFORMANCE METRICS")
    print("-"*90)
    print(f"{'Method':<20} {'R²':<10} {'RMSE':<10} {'MAE':<10} {'Competitive?':<15}")
    print("-"*90)
    
    for name, metrics in results_dict.items():
        r2 = metrics['r2']
        rmse = metrics['rmse']
        mae = metrics['mae']
        
        # Check if competitive (within 10% of MLP R²)
        if name == 'MLP':
            competitive = "Baseline"
        else:
            gap = mlp_r2 - r2
            if gap <= 0.10:
                competitive = "✓ Yes"
            elif gap <= 0.15:
                competitive = "⚠ Marginal"
            else:
                competitive = "✗ No"
        
        print(f"{name:<20} {r2:<10.4f} {rmse:<10.4f} {mae:<10.4f} {competitive:<15}")
    
    # 2. Efficiency Metrics
    print("\n⚡ 2. EFFICIENCY METRICS (Our Advantage!)")
    print("-"*90)
    print(f"{'Method':<20} {'Params':<12} {'Efficiency':<15} {'vs MLP':<15}")
    print("-"*90)
    
    for name, metrics in results_dict.items():
        params = metrics['n_params']
        eff = metrics['param_efficiency']
        
        if name == 'MLP':
            speedup = "Baseline"
        else:
            speedup_ratio = eff / mlp_eff if mlp_eff > 0 else 0
            speedup = f"{speedup_ratio:.1f}× better"
        
        print(f"{name:<20} {params:<12,} {eff:<15.4f} {speedup:<15}")
    
    # 3. Interpretability
    print("\n🔍 3. INTERPRETABILITY (Our Biggest Advantage!)")
    print("-"*90)
    print(f"{'Method':<20} {'Terms':<12} {'Interp Score':<15} {'Human Readable?':<20}")
    print("-"*90)
    
    for name, metrics in results_dict.items():
        terms = metrics['n_terms']
        interp = metrics['interpretability']
        
        if terms is None:
            terms_str = "N/A"
            readable = "✗ Black box"
        else:
            terms_str = str(terms)
            if terms <= 5:
                readable = "✓✓✓ Crystal clear"
            elif terms <= 10:
                readable = "✓✓ Very clear"
            elif terms <= 20:
                readable = "✓ Readable"
            else:
                readable = "⚠ Complex"
        
        print(f"{name:<20} {terms_str:<12} {interp:<15.4f} {readable:<20}")
    
    # 4. Composite Score
    print("\n🏆 4. COMPOSITE SCORE (40% perf + 30% eff + 30% interp)")
    print("-"*90)
    print(f"{'Method':<20} {'Composite':<15} {'Rank':<10}")
    print("-"*90)
    
    # Sort by composite score
    sorted_methods = sorted(results_dict.items(), key=lambda x: x[1]['composite_score'], reverse=True)
    
    for rank, (name, metrics) in enumerate(sorted_methods, 1):
        comp = metrics['composite_score']
        rank_str = f"#{rank}"
        print(f"{name:<20} {comp:<15.2f} {rank_str:<10}")
    
    # 5. Winner Analysis
    print("\n🎯 5. WINNER ANALYSIS")
    print("-"*90)
    
    best_perf = max(results_dict.items(), key=lambda x: x[1]['r2'])
    best_eff = max(results_dict.items(), key=lambda x: x[1]['param_efficiency'])
    best_interp = max(results_dict.items(), key=lambda x: x[1]['interpretability'])
    best_comp = sorted_methods[0]
    
    print(f"Best Performance:      {best_perf[0]:<20} (R²={best_perf[1]['r2']:.4f})")
    print(f"Best Efficiency:       {best_eff[0]:<20} (Score={best_eff[1]['param_efficiency']:.4f})")
    print(f"Best Interpretability: {best_interp[0]:<20} (Score={best_interp[1]['interpretability']:.4f})")
    print(f"Best Composite:        {best_comp[0]:<20} (Score={best_comp[1]['composite_score']:.2f})")
    
    # 6. Key Takeaways
    print("\n" + "="*90)
    print("💡 KEY TAKEAWAYS")
    print("="*90)
    
    # Find VSR methods
    vsr_methods = [name for name in results_dict.keys() if 'VSR' in name or 'RL' in name]
    
    if vsr_methods and 'MLP' in results_dict:
        # Use best VSR method
        best_vsr_name = max(vsr_methods, key=lambda n: results_dict[n]['r2'])
        vsr_metrics = results_dict[best_vsr_name]
        mlp_metrics = results_dict['MLP']
        
        # Calculate gaps
        r2_gap = mlp_metrics['r2'] - vsr_metrics['r2']
        r2_pct = (vsr_metrics['r2'] / mlp_metrics['r2']) * 100
        eff_gain = vsr_metrics['param_efficiency'] / mlp_metrics['param_efficiency']
        param_reduction = (1 - vsr_metrics['n_params'] / mlp_metrics['n_params']) * 100
        
        print(f"✓ {best_vsr_name} achieves {r2_pct:.1f}% of MLP's R²")
        print(f"✓ {best_vsr_name} is {eff_gain:.1f}× more parameter-efficient")
        print(f"✓ {best_vsr_name} uses {param_reduction:.2f}% fewer parameters")
        print(f"✓ {best_vsr_name} provides interpretable formulas (MLP is black box)")
        print(f"\n📝 Trade-off: Sacrifice {abs(r2_gap)*100:.1f}% R² for {param_reduction:.1f}% parameter reduction + full interpretability")
    
    print("="*90 + "\n")


def measure_inference_time(model, X_test, n_runs: int = 100):
    """
    Measure inference time for a model.
    
    Args:
        model: Model with predict() method
        X_test: Test data
        n_runs: Number of runs for averaging
    
    Returns:
        Average inference time in seconds
    """
    times = []
    
    for _ in range(n_runs):
        start = time.time()
        _ = model.predict(X_test)
        end = time.time()
        times.append(end - start)
    
    return np.mean(times)


# Example usage
if __name__ == "__main__":
    # Demo with synthetic data
    np.random.seed(42)
    y_true = np.random.randn(100)
    
    # MLP predictions (best performance)
    y_mlp = y_true + np.random.randn(100) * 0.1
    
    # VSR predictions (slightly worse but simpler)
    y_vsr = y_true + np.random.randn(100) * 0.15
    
    # Compute metrics
    mlp_metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_mlp,
        model_name="MLP",
        n_params=9473,
        n_terms=None
    )
    
    vsr_rl_metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_vsr,
        model_name="VSR+RL",
        n_params=7,
        n_terms=7
    )
    
    # Compare
    compare_methods({
        'MLP': mlp_metrics,
        'VSR+RL': vsr_rl_metrics
    }, dataset_name="Demo")
