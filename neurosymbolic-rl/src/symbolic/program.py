"""
Symbolic Program: Parse and execute Polish Notation expressions.

CRITICAL REQUIREMENT: Execution MUST be vectorized (batch processing).
The entire batch of latent vectors z is processed simultaneously.

Example Expression (Polish Notation):
    [add, mul, z0, z1, sin, z2]
    Represents: (z0 * z1) + sin(z2)

Execution Strategy:
- Use stack-based evaluation
- All operations are vectorized tensor ops
- Input: z [batch_size, latent_dim]
- Output: result [batch_size, 1] or [batch_size, latent_dim]
"""

import torch
from typing import List, Tuple
from src.symbolic.operators import OperatorLibrary, TokenVocabulary


class SymbolicProgram:
    """
    Represents and executes a symbolic expression in Polish Notation.
    
    Key Methods:
    - parse(token_sequence): Convert token indices to executable program
    - execute(z): Run program on batch of latent vectors (VECTORIZED)
    - validate(): Check if expression is syntactically valid
    """
    
    def __init__(
        self,
        token_sequence: List[int],
        vocabulary: TokenVocabulary,
        device: str = "cuda"
    ):
        """
        Args:
            token_sequence: List of token indices (Polish Notation)
            vocabulary: Token vocabulary for decoding
            device: Device for tensor operations
        """
        self.token_sequence = token_sequence
        self.vocabulary = vocabulary
        self.device = device
        self.operators = OperatorLibrary()
        
        # Parse tokens to get operator names and variable indices
        self.parsed_expression = self._parse_tokens()
    
    def _parse_tokens(self) -> List[Tuple[str, str]]:
        """
        Parse token indices into (token_type, token_value) pairs.
        
        Returns:
            List of (type, value) where:
            - type: "operator", "variable", or "special"
            - value: operator name, variable index, or special token
        """
        parsed = []
        for token_idx in self.token_sequence:
            token_str = self.vocabulary.decode(token_idx)
            
            if token_str in self.vocabulary.operators:
                parsed.append(("operator", token_str))
            elif token_str in self.vocabulary.variables:
                # Extract index from z0, z1, etc.
                parsed.append(("variable", token_str[1:]))
            elif token_str in self.vocabulary.special_tokens:
                parsed.append(("special", token_str))
            else:
                # Fallback or error
                parsed.append(("unknown", token_str))
        return parsed
    
    def execute(self, z: torch.Tensor) -> torch.Tensor:
        """
        Execute symbolic expression on batch of latent vectors.
        
        CRITICAL: This MUST be vectorized. All operations happen
        on the entire batch simultaneously.
        
        Algorithm (Stack-based Polish Notation evaluation):
        1. Initialize empty stack
        2. Iterate tokens in REVERSE order
        3. If variable (z[i]): push z[:, i] to stack
        4. If operator: pop k operands, apply op, push result
        5. Final stack should have 1 element (the result)
        
        Args:
            z: Latent vectors [batch_size, latent_dim]
        
        Returns:
            result: Expression output [batch_size, 1]
        """
        batch_size = z.shape[0]
        stack = []
        
        valid_tokens = [t for t in self.parsed_expression if t[0] not in ("special", "unknown")]
        
        for token_type, token_value in valid_tokens:
            if token_type == "variable":
                var_idx = int(token_value)  # z0 -> 0
                val = z[:, var_idx:var_idx+1] # [batch_size, 1]
                stack.append(val)
            
            elif token_type == "operator":
                op_name = token_value
                arity = self.operators.get_arity(op_name)
                
                if len(stack) < arity:
                    # Invalid expression (stack underflow)
                    raise ValueError(f"Stack Underflow: op={op_name}, stack={len(stack)}")
                
                op_func = self.operators.get_operator_dict()[op_name]
                if arity == 1:
                    a = stack.pop()
                    result = op_func(a)
                elif arity == 2:
                    b = stack.pop()
                    a = stack.pop()
                    result = op_func(a, b)
                else:
                    operands = [stack.pop() for _ in range(arity)]
                    result = op_func(*reversed(operands))
                stack.append(result)
        
        if len(stack) != 1:
            # Invalid expression (stack should have exactly 1 item)
            raise ValueError(f"Invalid Stack Size: {len(stack)} (expected 1)")
            
        return stack[0]
    
    def validate(self) -> bool:
        """
        Check if expression is syntactically valid.
        
        Validation rules:
        - Stack should never be empty when popping
        - Final stack should have exactly 1 element
        """
        stack_depth = 0
        valid_tokens = [t for t in self.parsed_expression if t[0] not in ("special", "unknown")]
        
        try:
            for token_type, token_value in valid_tokens:
                if token_type == "variable":
                    stack_depth += 1
                elif token_type == "operator":
                    op_name = token_value
                    arity = self.operators.get_arity(op_name)
                    
                    if stack_depth < arity:
                        return False

                    # Pop arity items, push 1 result
                    stack_depth -= (arity - 1)
            
            return stack_depth == 1
        except Exception:
            return False

    def to_string(self) -> str:
        """
        Convert expression to human-readable string.
        
        Example: [add, mul, z0, z1, sin, z2] → "((z0 * z1) + sin(z2))"
        """
        stack = []
        arity_map = self.operators.get_arity_map()
        
        valid_tokens = [t for t in self.parsed_expression if t[0] not in ("special", "unknown")]
        
        try:
            for token_type, value in valid_tokens:
                if token_type == "variable":
                    stack.append(f"z{value}")
                elif token_type == "operator":
                    arity = arity_map.get(value, 2)
                    if len(stack) < arity:
                        return f"Invalid (Underflow: op={value}, stack={stack})"
                    
                    if arity == 2:
                        b = stack.pop()
                        a = stack.pop()
                        infix_sym = {
                            "add": "+", "sub": "-", "mul": "*", "div": "/",
                        }
                        if value in infix_sym:
                            stack.append(f"({a} {infix_sym[value]} {b})")
                        else:
                            stack.append(f"{value}({a}, {b})")
                    elif arity == 1:
                        a = stack.pop()
                        stack.append(f"{value}({a})")
                    else:
                        operands = [stack.pop() for _ in range(arity)]
                        operands.reverse()
                        stack.append(f"{value}({', '.join(operands)})")
                else:
                    # Ignore special tokens or handle them
                    pass
            
            if len(stack) != 1:
                return f"Invalid (Stack size: {len(stack)})"
            return stack[0]
        except Exception as e:
            return f"Error: {str(e)}"
