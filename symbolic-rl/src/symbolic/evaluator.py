import torch
import torch.nn as nn
from typing import Tuple, Dict
from src.symbolic.program import SymbolicProgram


class ProgramEvaluator:
    """
    Evaluates symbolic programs using ONLY the program output.
    
    CRITICAL DESIGN:
    - Classifier sees ONLY prog_output (1D scalar)
    - Classifier NEVER sees original z features
    - Forces programs to be actually useful
    - Prevents cheating: "ignore program, use z directly"
    
    This aligns with the VSR paper's goal:
    - Find symbolic formula f(z) such that f(z) alone is sufficient for classification
    """
    def __init__(self, num_classes: int = 10, latent_dim: int = 10, device: str = "cpu"):
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self._eval_count = 0

    def _make_classifier(self) -> nn.Module:
        """
        Create simple LINEAR classifier.
        
        Input: 1D (program output only)
        Output: 10D (class logits)
        
        No hidden layers = no overfitting on small data.
        """
        return nn.Linear(1, self.num_classes).to(self.device)

    def evaluate(
        self,
        program: SymbolicProgram,
        z: torch.Tensor,
        labels: torch.Tensor,
        update_classifier: bool = True,
        n_train_steps: int = 20  # Reduced since Linear is simpler
    ) -> Tuple[float, Dict]:
        """
        Evaluate program using ONLY its output.
        
        Flow:
        1. Execute program: f(z) → prog_output
        2. Train/Val split (70/30)
        3. Train Linear(1, 10) on prog_output ONLY
        4. Evaluate on independent val set
        5. Return val accuracy
        
        Key: z features are NEVER passed to classifier.
        """
        try:
            # 1. Execute program
            with torch.no_grad():
                prog_output = program.execute(z)

            prog_output = prog_output.to(self.device)
            labels = labels.to(self.device).view(-1).long()

            # Handle NaN/Inf
            prog_output = torch.nan_to_num(prog_output, nan=0.0, posinf=1e4, neginf=-1e4)
            prog_output = prog_output.view(-1, 1)

            # Normalize ONLY program output
            mean = prog_output.mean()
            std = prog_output.std()
            if std > 1e-5:
                prog_output_norm = (prog_output - mean) / (std + 1e-8)
            else:
                prog_output_norm = prog_output - mean

            # CRITICAL: features = ONLY prog_output (no z!)
            features = prog_output_norm.detach()  # [batch, 1]

            # 2. Train/Val Split (70/30)
            n_samples = features.size(0)
            n_train = int(0.7 * n_samples)
            
            indices = torch.randperm(n_samples, device=self.device)
            train_idx = indices[:n_train]
            val_idx = indices[n_train:]
            
            train_features = features[train_idx]
            train_labels = labels[train_idx]
            val_features = features[val_idx]
            val_labels = labels[val_idx]

            # 3. Train linear classifier on TRAIN set only
            if update_classifier and n_train > 0:
                classifier = self._make_classifier()
                optimizer = torch.optim.Adam(classifier.parameters(), lr=0.05)  # Higher LR for linear
                
                classifier.train()
                for step in range(n_train_steps):
                    optimizer.zero_grad()
                    logits = classifier(train_features)
                    loss = self.criterion(logits, train_labels)
                    loss.backward()
                    optimizer.step()
                    
                    # Early stop if loss converged
                    if step > 5 and loss.item() < 0.1:
                        break
            else:
                classifier = self._make_classifier()

            # 4. Evaluate on INDEPENDENT validation set
            classifier.eval()
            with torch.no_grad():
                if len(val_idx) > 0:
                    val_logits = classifier(val_features)
                    val_loss = self.criterion(val_logits, val_labels)
                    val_predictions = val_logits.argmax(dim=1)
                    accuracy = (val_predictions == val_labels).float().mean().item()
                else:
                    # Fallback if split fails (shouldn't happen)
                    val_logits = classifier(features)
                    val_loss = self.criterion(val_logits, labels)
                    val_predictions = val_logits.argmax(dim=1)
                    accuracy = (val_predictions == labels).float().mean().item()

            self._eval_count += 1
            return accuracy, {"accuracy": accuracy, "loss": val_loss.item()}

        except Exception as e:
            print(f"[Evaluator Error] {e}")
            import traceback
            traceback.print_exc()
            return 0.0, {"accuracy": 0.0, "error": str(e)}
