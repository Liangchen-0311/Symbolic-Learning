import torch
from typing import List
from src.symbolic.operators import TokenVocabulary, OperatorLibrary


class ActionMasker:
    def __init__(self, vocabulary: TokenVocabulary):
        self.vocab = vocabulary
        self.operators = OperatorLibrary()
        self.start_idx = vocabulary.encode("START")
        self.end_idx = vocabulary.encode("END")
        self.pad_idx = vocabulary.encode("PAD")
        self.unary_ops = []
        self.binary_ops = []
        self.ternary_ops = []
        for op_name in vocabulary.operators:
            idx = vocabulary.encode(op_name)
            arity = self.operators.get_arity(op_name)
            if arity == 1:
                self.unary_ops.append(idx)
            elif arity == 2:
                self.binary_ops.append(idx)
            elif arity == 3:
                self.ternary_ops.append(idx)
        self.var_indices = [vocabulary.encode(f"z{i}") for i in range(vocabulary.latent_dim)]

    def get_valid_actions(self, token_sequence: List[int]) -> torch.Tensor:
        vocab_size = len(self.vocab)
        mask = torch.zeros(vocab_size, dtype=torch.bool)

        clean_seq = [
            t for t in token_sequence
            if t not in [self.start_idx, self.end_idx, self.pad_idx]
        ]
        stack_depth = self._compute_stack_depth(clean_seq)

        for idx in self.var_indices:
            mask[idx] = True

        if stack_depth >= 1:
            for idx in self.unary_ops:
                mask[idx] = True

        if stack_depth >= 2:
            for idx in self.binary_ops:
                mask[idx] = True

        if stack_depth >= 3:
            for idx in self.ternary_ops:
                mask[idx] = True

        if stack_depth == 1:
            mask[self.end_idx] = True

        mask[self.start_idx] = False
        mask[self.pad_idx] = False

        return mask

    def _compute_stack_depth(self, token_sequence: List[int]) -> int:
        depth = 0
        for token_idx in token_sequence:
            token_name = self.vocab.decode(token_idx)
            if token_name.startswith("z"):
                depth += 1
            elif token_name in self.vocab.operators:
                arity = self.operators.get_arity(token_name)
                if depth < arity:
                    return -1
                depth -= (arity - 1)
        return depth
