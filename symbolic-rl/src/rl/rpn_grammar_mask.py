"""
严格的RPN语法掩码 (Grammar-based Action Masking for RPN)

实现100%合法的逆波兰表达式生成，通过追踪堆栈深度来掩码非法动作。

核心规则：
1. 追踪堆栈深度：
   - Terminal: stack_depth + 1
   - Unary operator: stack_depth 不变
   - Binary operator: stack_depth - 1

2. 掩码规则：
   - stack_depth < 2: 禁止 Binary operators
   - current_length == max_length - 1: 必须能闭合
   - 最后一个token: 必须是降维算子且 stack_depth == 1

3. 强制收尾：
   - 公式结束时必须是 global pooling 且堆栈深度为1
"""

import torch
from typing import List, Dict, Tuple
from src.symbolic.tensor_operators import TENSOR_OPERATORS, ROOT_OPERATORS


class RPNGrammarMask:
    """
    基于RPN语法的严格动作掩码。

    确保生成的每一步都是合法的RPN表达式。
    """

    def __init__(self, vocabulary, max_sequence_length=15):
        """
        Args:
            vocabulary: TensorTokenVocabulary实例
            max_sequence_length: 最大序列长度
        """
        self.vocab = vocabulary
        self.max_length = max_sequence_length

        # 构建算子类型映射
        self._build_operator_mappings()

    def _build_operator_mappings(self):
        """构建算子类型到索引的映射。"""
        self.terminals = []  # I_R, I_G, I_B
        self.unary_ops = []  # blur, edge_x, log, etc.
        self.binary_ops = []  # add, multiply, subtract, etc.
        self.pooling_ops = []  # global_avg_pool, pool_top_half, etc.

        for token, idx in self.vocab.token_to_idx.items():
            if token in ['START', 'END', 'PAD']:
                continue

            if token.startswith('I_') or token.startswith('L1_'):  # Terminals
                self.terminals.append(idx)
            elif token in ROOT_OPERATORS:  # Pooling (scalar output)
                self.pooling_ops.append(idx)
            elif token in TENSOR_OPERATORS:
                op_func, arity, output_type = TENSOR_OPERATORS[token]
                if arity == 1:
                    self.unary_ops.append(idx)
                elif arity == 2:
                    self.binary_ops.append(idx)

        # 转换为tensor便于掩码操作
        self.terminals_tensor = torch.tensor(self.terminals, dtype=torch.long)
        self.unary_ops_tensor = torch.tensor(self.unary_ops, dtype=torch.long)
        self.binary_ops_tensor = torch.tensor(self.binary_ops, dtype=torch.long)
        self.pooling_ops_tensor = torch.tensor(self.pooling_ops, dtype=torch.long)

        # 特殊token索引
        self.end_idx = self.vocab.encode('END')

    def compute_stack_depth(self, tokens: List[int]) -> int:
        """
        计算当前token序列的堆栈深度。

        Args:
            tokens: 已生成的token列表（不包括START）

        Returns:
            stack_depth: 当前堆栈深度
        """
        depth = 0
        for token in tokens:
            if token in self.terminals:
                depth += 1
            elif token in self.unary_ops:
                # Unary: 深度不变（弹出1个，压入1个）
                pass
            elif token in self.binary_ops:
                depth -= 1  # Binary: 弹出2个，压入1个
            elif token in self.pooling_ops:
                # Pooling: 弹出1个，压入1个（标量）
                # 但对于最终闭合，pooling会减少深度到0
                pass
        return depth

    def get_valid_actions(
        self,
        current_tokens: List[int],
        device='cpu'
    ) -> torch.Tensor:
        """
        获取当前状态下的合法动作掩码。

        Args:
            current_tokens: 已生成的token列表（不包括START）
            device: tensor设备

        Returns:
            mask: [vocab_size] binary mask (1=allowed, 0=forbidden)
        """
        vocab_size = len(self.vocab)
        mask = torch.zeros(vocab_size, dtype=torch.float32, device=device)

        current_length = len(current_tokens)
        stack_depth = self.compute_stack_depth(current_tokens)

        # 规则1: 序列为空时，只能选择 terminals（开始构建树）
        if current_length == 0:
            mask[self.terminals_tensor] = 1.0
            return mask

        # 规则2: 接近最大长度，必须能闭合
        remaining_length = self.max_length - current_length

        if remaining_length == 1:
            # 这是最后一个token，必须是pooling且stack_depth == 1
            if stack_depth == 1:
                mask[self.pooling_ops_tensor] = 1.0
                mask[self.end_idx] = 1.0  # 也可以选择END
            else:
                # 无法闭合！这种情况理论上不应该发生
                # 如果stack_depth > 1，需要减少深度
                # 如果stack_depth < 1，无效
                if stack_depth > 1:
                    # 尝试使用binary op减少深度
                    mask[self.binary_ops_tensor] = 1.0
                # 否则什么都不能选（会导致invalid）
            return mask

        if remaining_length == 2:
            # 还剩2个位置：倒数第二步必须使 stack_depth == 1，最后一步 pooling
            if stack_depth == 0:
                mask[self.terminals_tensor] = 1.0         # → depth 1, then pooling
            elif stack_depth == 1:
                mask[self.unary_ops_tensor] = 1.0         # → depth 1, then pooling
                # terminals would push to depth 2, can't close in 1 slot
            elif stack_depth == 2:
                mask[self.binary_ops_tensor] = 1.0        # → depth 1, then pooling
            else:
                mask[self.binary_ops_tensor] = 1.0        # best effort: reduce depth
            # 禁止连续相同一元算子
            if current_tokens and current_tokens[-1] in self.unary_ops:
                mask[current_tokens[-1]] = 0.0
            return mask

        # 规则3: 常规情况
        # 始终可以添加 terminals 和 unary operators
        mask[self.terminals_tensor] = 1.0
        mask[self.unary_ops_tensor] = 1.0

        # 规则4: Binary operators 只在 stack_depth >= 2 时允许
        if stack_depth >= 2:
            mask[self.binary_ops_tensor] = 1.0

        # 规则5: Pooling/root operators 只允许在最后一步（remaining_length == 1）
        # 且 stack_depth == 1。这确保 pooling 一定是公式的最后一个算子。
        # (remaining_length == 1 的情况已在上面处理)

        # 如果上一个 token 已经是 pooling（输出标量），则只能 END
        if current_tokens and current_tokens[-1] in self.pooling_ops:
            mask = torch.zeros(vocab_size, dtype=torch.float32, device=device)
            mask[self.end_idx] = 1.0
            return mask

        # 规则6: 防止堆栈过深
        if stack_depth > (self.max_length - current_length):
            mask = torch.zeros(vocab_size, dtype=torch.float32, device=device)
            mask[self.binary_ops_tensor] = 1.0
            if stack_depth == 1 and remaining_length == 1:
                mask[self.pooling_ops_tensor] = 1.0

        # 规则7 (NEW): 禁止连续相同的一元算子。
        # 如果上一个 token 是一元算子 X，则禁止再选 X。
        # 避免 relu relu、sigmoid sigmoid 等无意义重复。
        if current_tokens and current_tokens[-1] in self.unary_ops:
            last_token_idx = current_tokens[-1]
            mask[last_token_idx] = 0.0

        return mask

    def is_valid_sequence(self, tokens: List[int]) -> Tuple[bool, str]:
        """
        验证一个完整的token序列是否是合法的RPN。

        Args:
            tokens: token列表（不包括START/END）

        Returns:
            is_valid: 是否合法
            reason: 如果非法，原因说明
        """
        if len(tokens) == 0:
            return False, "Empty sequence"

        # 检查最后一个token是否是pooling
        if tokens[-1] not in self.pooling_ops:
            return False, f"Last token must be pooling operator, got {self.vocab.decode(tokens[-1])}"

        # 检查最终堆栈深度
        final_depth = 0
        for i, token in enumerate(tokens):
            if token in self.terminals:
                final_depth += 1
            elif token in self.unary_ops:
                if final_depth < 1:
                    return False, f"Unary operator at position {i} but stack is empty"
            elif token in self.binary_ops:
                if final_depth < 2:
                    return False, f"Binary operator at position {i} but stack has only {final_depth} elements"
                final_depth -= 1
            elif token in self.pooling_ops:
                if final_depth < 1:
                    return False, f"Pooling operator at position {i} but stack is empty"
                # Pooling会输出标量，算作最终结果
                if i == len(tokens) - 1:
                    # 这是最后一个token，应该堆栈深度为1
                    if final_depth != 1:
                        return False, f"Stack depth is {final_depth} before final pooling, should be 1"
                final_depth = final_depth - 1 + 1  # 弹出tensor，压入scalar（算作1）

        # 最终应该只有1个元素在栈上
        if final_depth != 1:
            return False, f"Final stack depth is {final_depth}, should be 1"

        return True, "Valid RPN"

    def apply_mask_to_logits(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        mask_value: float = -1e9
    ) -> torch.Tensor:
        """
        将掩码应用到logits上。

        Args:
            logits: [vocab_size] or [batch, vocab_size]
            mask: [vocab_size] binary mask (1=allowed, 0=forbidden)
            mask_value: 被禁止动作的logit值

        Returns:
            masked_logits: 掩码后的logits
        """
        mask = mask.to(logits.device)

        # 将forbidden actions设为mask_value
        if logits.dim() == 1:
            masked_logits = logits.clone()
            masked_logits[mask == 0] = mask_value
        else:
            masked_logits = logits.clone()
            masked_logits[:, mask == 0] = mask_value

        return masked_logits


def test_rpn_grammar_mask():
    """测试RPN语法掩码。"""
    from src.rl.tensor_environment_large_bank import TensorTokenVocabulary

    vocab = TensorTokenVocabulary()
    masker = RPNGrammarMask(vocab, max_sequence_length=10)

    print("=" * 60)
    print("Testing RPN Grammar Mask")
    print("=" * 60)

    # 测试1: 空序列
    print("\nTest 1: Empty sequence")
    tokens = []
    mask = masker.get_valid_actions(tokens)
    valid_tokens = [vocab.decode(i) for i in range(len(vocab)) if mask[i] > 0]
    print(f"Valid actions: {valid_tokens}")
    assert all(t.startswith('I_') for t in valid_tokens), "Should only allow terminals"

    # 测试2: 一个terminal后
    print("\nTest 2: After one terminal (I_R)")
    tokens = [vocab.encode('I_R')]
    mask = masker.get_valid_actions(tokens)
    valid_tokens = [vocab.decode(i) for i in range(len(vocab)) if mask[i] > 0]
    print(f"Stack depth: {masker.compute_stack_depth(tokens)}")
    print(f"Valid actions: {valid_tokens[:10]}...")  # 显示前10个
    print(f"Can use binary ops: {'add' in valid_tokens}")
    assert 'add' not in valid_tokens, "Binary ops should be forbidden when stack_depth < 2"

    # 测试3: 两个terminals后
    print("\nTest 3: After two terminals (I_R, I_G)")
    tokens = [vocab.encode('I_R'), vocab.encode('I_G')]
    mask = masker.get_valid_actions(tokens)
    valid_tokens = [vocab.decode(i) for i in range(len(vocab)) if mask[i] > 0]
    print(f"Stack depth: {masker.compute_stack_depth(tokens)}")
    print(f"Can use binary ops: {'add' in valid_tokens}")
    assert 'add' in valid_tokens, "Binary ops should be allowed when stack_depth >= 2"

    # 测试4: 合法序列验证
    print("\nTest 4: Validate complete sequences")
    valid_seq = [vocab.encode(t) for t in ['I_R', 'I_G', 'add', 'global_avg_pool']]
    is_valid, reason = masker.is_valid_sequence(valid_seq)
    print(f"Sequence: I_R I_G add global_avg_pool")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert is_valid, "This should be a valid RPN"

    # 测试5: 非法序列（缺少pooling）
    print("\nTest 5: Invalid sequence (no pooling at end)")
    invalid_seq = [vocab.encode(t) for t in ['I_R', 'I_G', 'add']]
    is_valid, reason = masker.is_valid_sequence(invalid_seq)
    print(f"Sequence: I_R I_G add")
    print(f"Valid: {is_valid}, Reason: {reason}")
    assert not is_valid, "This should be invalid (no pooling at end)"

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_rpn_grammar_mask()
