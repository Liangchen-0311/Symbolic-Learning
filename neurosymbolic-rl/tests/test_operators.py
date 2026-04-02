
import torch
import pytest
from src.symbolic.operators import OperatorLibrary, TokenVocabulary
from src.symbolic.program import SymbolicProgram

class TestOperators:
    def test_add_vectorized(self):
        batch_size = 128
        x = torch.ones(batch_size, 1)
        y = torch.ones(batch_size, 1) * 2
        
        result = OperatorLibrary.add(x, y)
        
        assert result.shape == (batch_size, 1)
        assert torch.all(result == 3)
        
    def test_div_protected(self):
        x = torch.tensor([[10.0], [10.0]])
        y = torch.tensor([[2.0], [0.0]]) # Division by zero case
        
        result = OperatorLibrary.div(x, y)
        
        # Check normal division
        assert torch.abs(result[0] - 5.0) < 1e-6
        # Check protected division (should not be NaN or Inf, should be large number or handled)
        # The implementation uses y + epsilon
        assert not torch.isnan(result[1])
        assert not torch.isinf(result[1])
        
    def test_all_operators_vectorized(self):
        batch_size = 32
        x = torch.randn(batch_size, 1)
        y = torch.randn(batch_size, 1)
        
        z = torch.randn(batch_size, 1)

        ops = OperatorLibrary.get_operator_dict()
        for name, func in ops.items():
            arity = OperatorLibrary.get_arity(name)
            if arity == 1:
                res = func(x)
            elif arity == 2:
                res = func(x, y)
            elif arity == 3:
                res = func(x, y, z)
            assert res.shape == (batch_size, 1)

class TestSymbolicProgram:
    def setup_method(self):
        self.vocab = TokenVocabulary(latent_dim=10)
        
    def test_simple_execution(self):
        # Expression: z0 + z1
        # Polish: [add, z0, z1]
        tokens = [
            self.vocab.encode('add'),
            self.vocab.encode('z0'),
            self.vocab.encode('z1')
        ]
        
        program = SymbolicProgram(tokens, self.vocab)
        
        batch_size = 10
        latent_dim = 10
        z = torch.zeros(batch_size, latent_dim)
        z[:, 0] = 2.0 # z0
        z[:, 1] = 3.0 # z1
        
        result = program.execute(z)
        
        assert result.shape == (batch_size, 1)
        assert torch.all(result == 5.0)

    def test_complex_execution(self):
        # Expression: (z0 * z1) + sin(z2)
        # Polish: [add, mul, z0, z1, sin, z2]
        tokens = [
            self.vocab.encode('add'),
            self.vocab.encode('mul'),
            self.vocab.encode('z0'),
            self.vocab.encode('z1'),
            self.vocab.encode('sin'),
            self.vocab.encode('z2')
        ]
        
        program = SymbolicProgram(tokens, self.vocab)
        
        batch_size = 5
        z = torch.zeros(batch_size, 10)
        z[:, 0] = 2.0
        z[:, 1] = 3.0
        z[:, 2] = 0.0 # sin(0) = 0
        
        # Expected: (2*3) + sin(0) = 6 + 0 = 6
        result = program.execute(z)
        
        assert torch.allclose(result, torch.tensor(6.0), atol=1e-6)

    def test_invalid_expression(self):
        # Invalid: [add, z0] (missing operand)
        tokens = [
            self.vocab.encode('add'),
            self.vocab.encode('z0')
        ]
        program = SymbolicProgram(tokens, self.vocab)
        z = torch.randn(5, 10)
        result = program.execute(z)
        
        # Should return zeros for invalid expression
        assert torch.all(result == 0)

