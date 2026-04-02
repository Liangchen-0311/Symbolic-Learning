"""
Reward shaping utilities for RL training.

Provides flexible reward computation strategies beyond simple accuracy.
"""

import torch
from typing import Dict


class RewardShaper:
    """
    Computes shaped rewards for symbolic expressions.
    
    Reward Components:
    1. Task Performance (accuracy/loss)
    2. Validity (syntactic correctness)
    3. Parsimony (expression length)
    4. Diversity (novelty bonus)
    """
    
    def __init__(
        self,
        accuracy_weight: float = 1.0,
        validity_weight: float = 0.5,
        parsimony_weight: float = 0.01,
        diversity_weight: float = 0.1
    ):
        self.accuracy_weight = accuracy_weight
        self.validity_weight = validity_weight
        self.parsimony_weight = parsimony_weight
        self.diversity_weight = diversity_weight
        
        # Track seen expressions for diversity bonus
        self.seen_expressions = set()
    
    def compute_reward(
        self,
        accuracy: float,
        is_valid: bool,
        expression_length: int,
        expression_hash: str
    ) -> float:
        """
        Compute total shaped reward.
        
        Args:
            accuracy: Classification accuracy [0, 1]
            is_valid: Whether expression is syntactically valid
            expression_length: Number of tokens in expression
            expression_hash: String representation for diversity
        
        Returns:
            total_reward: Shaped reward
        """
        reward = 0.0
        
        # Task performance
        reward += self.accuracy_weight * accuracy
        
        # Validity
        if is_valid:
            reward += self.validity_weight
        else:
            reward -= self.validity_weight
        
        # Parsimony (prefer shorter expressions)
        parsimony_penalty = -self.parsimony_weight * expression_length
        reward += parsimony_penalty
        
        # Diversity (bonus for novel expressions)
        if expression_hash not in self.seen_expressions:
            reward += self.diversity_weight
            self.seen_expressions.add(expression_hash)
        
        return reward