"""
LASSO (L1-regularized) Linear Classifier for VSR features.
"""

import torch
import torch.nn as nn
import torch.optim as optim


class LASSOLinearClassifier(nn.Module):
    """
    Simple linear classifier with L1 regularization (LASSO).

    Input: [batch, num_features]
    Output: [batch, num_classes]
    """

    def __init__(self, num_features, num_classes=10):
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.linear(x)

    def l1_penalty(self):
        """Compute L1 penalty on weights."""
        return torch.norm(self.linear.weight, p=1)

    def l2_penalty(self):
        """Compute L2 penalty on weights."""
        return torch.sum(self.linear.weight ** 2)

    def count_active_features(self, threshold=1e-4):
        """
        Count features with non-zero weights.

        A feature is "active" if any class has non-zero weight for it.
        """
        active = (torch.abs(self.linear.weight) > threshold).any(dim=0)
        return active.sum().item()


def train_lasso_classifier(
    features_tensor,
    targets,
    num_classes=10,
    l1_lambda=0.01,
    l2_lambda=0.0,
    epochs=50,
    lr=0.01,
    device='cuda'
):
    """
    Train Elastic Net classifier on extracted features.

    Args:
        features_tensor: [batch, num_features] from formulas
        targets: [batch] class labels
        num_classes: Number of classes
        l1_lambda: L1 regularization strength (sparsity)
        l2_lambda: L2 regularization strength (stability)
        epochs: Training epochs
        lr: Learning rate
        device: Device

    Returns:
        accuracy: Validation accuracy
        active_features: Number of active features
        model: Trained model
    """
    num_features = features_tensor.shape[1]

    # Create model
    model = LASSOLinearClassifier(num_features, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features_tensor)

        # Classification loss
        cls_loss = criterion(outputs, targets)

        # L1 penalty (LASSO - sparsity)
        l1_penalty = model.l1_penalty()

        # L2 penalty (Ridge - stability)
        l2_penalty = model.l2_penalty()

        # Total loss (Elastic Net)
        loss = cls_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

        # Backward pass
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == targets).float().mean().item()

        # Count active features
        active_features = model.count_active_features(threshold=1e-4)

    return accuracy, active_features, model


def train_lasso_classifier_with_selection(
    features_tensor,
    targets,
    num_classes=10,
    l1_lambda=0.5,
    l2_lambda=0.0,
    epochs=200,
    lr=0.01,
    device='cuda',
    threshold=1e-3
):
    """
    Train Elastic Net classifier with automatic feature selection.

    This is the STRONG regularization version for large feature banks.
    It automatically prunes redundant features and returns
    only the active (selected) feature indices.

    Args:
        features_tensor: [batch, num_features] from formulas
        targets: [batch] class labels
        num_classes: Number of classes
        l1_lambda: Strong L1 regularization (default 0.5, vs 0.01)
        l2_lambda: L2 regularization strength (stability)
        epochs: Training epochs (default 200 for strong convergence)
        lr: Learning rate
        device: Device
        threshold: Threshold for considering a feature "active"

    Returns:
        accuracy: Classification accuracy
        active_indices: Tensor of selected feature indices
        selected_count: Number of features selected
        model: Trained model
    """
    num_features = features_tensor.shape[1]

    # Create model
    model = LASSOLinearClassifier(num_features, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features_tensor)

        # Classification loss
        cls_loss = criterion(outputs, targets)

        # Strong L1 penalty (LASSO - sparsity)
        l1_penalty = model.l1_penalty()

        # L2 penalty (Ridge - stability)
        l2_penalty = model.l2_penalty()

        # Total loss (Elastic Net)
        loss = cls_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

        # Backward pass
        loss.backward()
        optimizer.step()

    # Evaluate and select features
    model.eval()
    with torch.no_grad():
        outputs = model(features_tensor)
        preds = torch.argmax(outputs, dim=1)
        accuracy = (preds == targets).float().mean().item()

        # Find active features based on weight importance
        weights = model.linear.weight  # [num_classes, num_features]
        feature_importance = torch.abs(weights).sum(dim=0)  # [num_features]

        # Threshold-based selection
        active_mask = feature_importance > threshold
        active_indices = torch.where(active_mask)[0]
        selected_count = len(active_indices)

        print(f"  [LASSO Selection] {selected_count}/{num_features} features selected ({selected_count/num_features*100:.1f}%)")
        print(f"  [LASSO] Accuracy: {accuracy*100:.2f}%")

    return accuracy, active_indices, selected_count, model
