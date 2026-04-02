"""
Action masking for tensor-based VSR.

Rules:
1. Root node MUST be a pooling operator (global_avg_pool, etc.)
2. Non-root nodes can be any tensor operator
3. Terminals are I_R, I_G, I_B
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.symbolic.tensor_operators import ROOT_OPERATORS, TENSOR_OPERATORS


def create_action_mask(
    current_depth,
    max_depth,
    is_root,
    vocab_size,
    root_op_indices,
    tensor_op_indices,
    terminal_indices
):
    """
    Create action mask for current step.

    Args:
        current_depth: Current tree depth
        max_depth: Maximum allowed depth
        is_root: Boolean, is this the root node?
        vocab_size: Total vocabulary size
        root_op_indices: Indices of root-only operators
        tensor_op_indices: Indices of tensor operators
        terminal_indices: Indices of terminals (I_R, I_G, I_B)

    Returns:
        mask: [vocab_size] binary mask (1=allowed, 0=forbidden)
    """
    mask = torch.zeros(vocab_size)

    if is_root:
        # Root node: MUST be pooling operator
        mask[root_op_indices] = 1.0
    elif current_depth >= max_depth:
        # Max depth: MUST be terminal
        mask[terminal_indices] = 1.0
    else:
        # Internal node: can be tensor operator or terminal
        mask[tensor_op_indices] = 1.0
        mask[terminal_indices] = 1.0

    return mask


class TensorActionMasker:
    """
    Action masking for tensor-based VSR.

    Ensures:
    - Root node is always a pooling operator (converts [B,H,W] → [B])
    - Internal nodes are tensor operators or terminals
    - Terminals are only used when needed
    """

    def __init__(self, vocab, max_depth=5):
        """
        Args:
            vocab: TokenVocabulary instance
            max_depth: Maximum tree depth
        """
        self.vocab = vocab
        self.max_depth = max_depth

        # Build operator type indices
        self._build_operator_indices()

    def _build_operator_indices(self):
        """Build indices for different operator types."""
        self.root_op_indices = []
        self.tensor_op_indices = []
        self.terminal_indices = []

        for token, idx in self.vocab.token_to_idx.items():
            if token in ROOT_OPERATORS:
                self.root_op_indices.append(idx)
            elif token in TENSOR_OPERATORS:
                _, _, output_type = TENSOR_OPERATORS[token]
                if output_type == 'tensor':
                    self.tensor_op_indices.append(idx)
            elif token.startswith('I_'):  # Terminals: I_R, I_G, I_B
                self.terminal_indices.append(idx)

        # Convert to tensors
        self.root_op_indices = torch.tensor(self.root_op_indices)
        self.tensor_op_indices = torch.tensor(self.tensor_op_indices)
        self.terminal_indices = torch.tensor(self.terminal_indices)

    def get_mask(self, depth, is_root=False):
        """
        Get action mask for current state.

        Args:
            depth: Current depth in tree
            is_root: Whether this is the root node

        Returns:
            mask: [vocab_size] binary mask
        """
        return create_action_mask(
            current_depth=depth,
            max_depth=self.max_depth,
            is_root=is_root,
            vocab_size=len(self.vocab),
            root_op_indices=self.root_op_indices,
            tensor_op_indices=self.tensor_op_indices,
            terminal_indices=self.terminal_indices
        )

    def apply_mask_to_logits(self, logits, mask):
        """
        Apply mask to logits.

        Args:
            logits: [batch, vocab_size] or [vocab_size]
            mask: [vocab_size] binary mask

        Returns:
            masked_logits: Same shape as logits
        """
        # Set forbidden actions to -inf
        mask = mask.to(logits.device)
        if logits.dim() == 1:
            masked_logits = logits + (mask - 1) * 1e9
        else:
            masked_logits = logits + (mask.unsqueeze(0) - 1) * 1e9
        return masked_logits


class MaskedPolicy(nn.Module):
    """Policy network with action masking."""

    def __init__(self, policy_net):
        """
        Args:
            policy_net: Base policy network that outputs logits
        """
        super().__init__()
        self.policy_net = policy_net

    def forward(self, state, mask):
        """
        Args:
            state: [batch, state_dim]
            mask: [batch, vocab_size] action mask

        Returns:
            action_probs: [batch, vocab_size] masked probabilities
        """
        # Get logits from policy network
        logits = self.policy_net(state)

        # Apply mask (set forbidden actions to -inf)
        mask = mask.to(logits.device)
        masked_logits = logits + (mask - 1) * 1e9

        # Softmax
        action_probs = F.softmax(masked_logits, dim=-1)

        return action_probs
