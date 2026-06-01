"""
Library of PyTorch-compatible tensor operators for symbolic expressions.

CRITICAL: All operators MUST support batch processing (vectorized execution).
NO scalar loops allowed.

Each operator takes batched tensors and returns batched results.
"""

import torch
from typing import Callable, Dict


class OperatorLibrary:
    """
    Registry of available operators for symbolic expressions.
    
    All operators operate on batched tensors:
    - Input: [batch_size, dim]
    - Output: [batch_size, dim] or [batch_size, 1]
    
    Operators are organized by arity:
    - Unary: sin, cos, exp, log, relu, sigmoid, square, sqrt, ...
    - Binary: add, sub, mul, div, pow, ...
    """
    
    @staticmethod
    def add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise addition (vectorized)"""
        return x + y
    
    @staticmethod
    def mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise multiplication (vectorized)"""
        return x * y
    
    @staticmethod
    def sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Element-wise subtraction (vectorized)"""
        return x - y
    
    @staticmethod
    def div(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Protected division (vectorized).
        Prevents division by zero using small epsilon.
        """
        epsilon = 1e-8
        return x / (y + epsilon)
    
    @staticmethod
    def sin(x: torch.Tensor) -> torch.Tensor:
        """Element-wise sine (vectorized)"""
        return torch.sin(x)
    
    @staticmethod
    def cos(x: torch.Tensor) -> torch.Tensor:
        """Element-wise cosine (vectorized)"""
        return torch.cos(x)
    
    @staticmethod
    def exp(x: torch.Tensor) -> torch.Tensor:
        """
        Protected exponential (vectorized).
        Clips input to prevent overflow.
        """
        return torch.exp(torch.clamp(x, -10, 10))
    
    @staticmethod
    def log(x: torch.Tensor) -> torch.Tensor:
        """
        Protected logarithm (vectorized).
        Uses absolute value + epsilon.
        """
        epsilon = 1e-8
        return torch.log(torch.abs(x) + epsilon)
    
    @staticmethod
    def relu(x: torch.Tensor) -> torch.Tensor:
        """ReLU activation (vectorized)"""
        return torch.relu(x)
    
    @staticmethod
    def square(x: torch.Tensor) -> torch.Tensor:
        """Element-wise square (vectorized)"""
        return x * x
    
    @staticmethod
    def sqrt(x: torch.Tensor) -> torch.Tensor:
        """Protected square root (vectorized)"""
        return torch.sqrt(torch.abs(x) + 1e-8)
    
    # ==================== Physics-aware Binary Operators ====================

    @staticmethod
    def mul_vars(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Cross-variable multiplication: x * y
        Essential for: E1*E2, m*v, F*d
        """
        return x * y

    @staticmethod
    def square_sum(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Sum of squares: x^2 + y^2
        Direct pattern for: Pythagorean theorem, invariant mass E1^2+E2^2
        """
        return x * x + y * y

    # ==================== Physics-aware Ternary Operator ====================

    @staticmethod
    def interaction_cos(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Three-way interaction with cosine: x * y * cos(z)
        Critical for particle physics: E1*E2*cos(theta)
        """
        return x * y * torch.cos(z)

    @classmethod
    def get_operator_dict(cls) -> Dict[str, Callable]:
        """
        Returns dictionary mapping operator names to functions.
        Used for token-to-function lookup during execution.
        """
        return {
            # Binary (arity=2)
            'add': cls.add,
            'mul': cls.mul,
            'sub': cls.sub,
            'div': cls.div,
            'mul_vars': cls.mul_vars,
            'square_sum': cls.square_sum,
            # Unary (arity=1)
            'sin': cls.sin,
            'cos': cls.cos,
            'exp': cls.exp,
            'log': cls.log,
            'relu': cls.relu,
            'square': cls.square,
            'sqrt': cls.sqrt,
            # Ternary (arity=3)
            'interaction_cos': cls.interaction_cos,
        }
    
    @classmethod
    def get_arity_map(cls) -> Dict[str, int]:
        """Returns dictionary mapping operator names to arity."""
        return {
            # Binary (2 arguments)
            'add': 2, 'sub': 2, 'mul': 2, 'div': 2,
            'mul_vars': 2, 'square_sum': 2,
            # Unary (1 argument)
            'sin': 1, 'cos': 1, 'exp': 1, 'log': 1,
            'relu': 1, 'square': 1, 'sqrt': 1,
            # Ternary (3 arguments)
            'interaction_cos': 3,
        }

    @classmethod
    def get_arity(cls, op_name: str) -> int:
        """
        Returns the arity (number of arguments) for an operator.
        """
        return cls.get_arity_map().get(op_name, 1)


# Token vocabulary management
class TokenVocabulary:
    """
    Manages token-to-index mapping for the policy agent.
    
    Token Types:
    - Special: START, END, PAD
    - Operators: add, mul, sin, cos, ...
    - Variables: z[0], z[1], ..., z[latent_dim-1]
    """
    
    def __init__(self, latent_dim: int = 10):
        self.latent_dim = latent_dim
        
        # Build vocabulary
        self.special_tokens = ['START', 'END', 'PAD']
        self.operators = list(OperatorLibrary.get_operator_dict().keys())
        self.variables = [f'z{i}' for i in range(latent_dim)]
        
        self.all_tokens = self.special_tokens + self.operators + self.variables
        
        # Create mappings
        self.token_to_idx = {token: idx for idx, token in enumerate(self.all_tokens)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
    
    def __len__(self) -> int:
        return len(self.all_tokens)
    
    def encode(self, token: str) -> int:
        """Convert token to index"""
        return self.token_to_idx[token]
    
    def decode(self, idx: int) -> str:
        """Convert index to token"""
        return self.idx_to_token[idx]