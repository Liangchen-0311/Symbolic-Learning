"""
Symbolic Backend: RL Policy Agent (LSTM/Transformer Controller).

This is the core RL component that generates mathematical expressions
in Polish Notation by sequentially selecting operators and operands.

KEY DIFFERENCE FROM SOURCE PAPER:
- Source paper uses Genetic Programming
- This implementation uses Reinforcement Learning (PPO)
- The agent is an LSTM/Transformer that outputs action probabilities
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class PolicyAgent(nn.Module):
    """
    RL Controller that generates symbolic expressions in Polish Notation.
    
    The agent operates as follows:
    1. Takes current state (previous token or empty)
    2. Outputs distribution over actions (operators + variable indices)
    3. Samples action using policy distribution
    4. Updates state and repeats until END token or max length
    
    Action Space:
    - Operators: [add, mul, sub, div, sin, cos, exp, log, relu, ...]
    - Variables: [z[0], z[1], ..., z[latent_dim-1]]
    - Special: [START, END]
    
    State Representation:
    - Embedding of previous token
    - Position encoding
    - (Optional) Context from partial expression
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_size: int = 256,
        num_layers: int = 2,
        model_type: str = "lstm",  # "lstm" or "transformer"
        dropout: float = 0.1
    ):
        """
        Args:
            vocab_size: Total number of tokens (operators + variables + special)
            embedding_dim: Dimension of token embeddings
            hidden_size: Hidden dimension for LSTM/Transformer
            num_layers: Number of layers
            model_type: "lstm" or "transformer"
            dropout: Dropout probability
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.model_type = model_type
        
        # Token embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        if model_type == "lstm":
            self.controller = nn.LSTM(
                embedding_dim,
                hidden_size,
                num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0
            )
        elif model_type == "transformer":
            # TODO: Implement transformer-based controller
            pass
        
        # Output head: maps hidden state to action probabilities
        self.action_head = nn.Linear(hidden_size, vocab_size)
        
        # Value head for PPO (estimates state value)
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(
        self,
        token_sequence: torch.Tensor,
        hidden_state: Optional[Tuple] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Forward pass for action selection.
        
        Args:
            token_sequence: Previous tokens [batch_size, seq_len]
            hidden_state: Hidden state from previous step (for LSTM)
        
        Returns:
            action_logits: Logits over actions [batch_size, vocab_size]
            value: State value estimate [batch_size, 1]
            new_hidden_state: Updated hidden state
        """
        # Embed tokens
        # [batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(token_sequence)
        
        # Pass through LSTM
        # output: [batch_size, seq_len, hidden_size]
        if self.model_type == "lstm":
            output, new_hidden_state = self.controller(embedded, hidden_state)
        else:
            raise NotImplementedError("Only LSTM is currently supported")
            
        # We only care about the last step for action prediction
        # [batch_size, hidden_size]
        last_step_output = output[:, -1, :]
        
        # Action logits
        action_logits = self.action_head(last_step_output)
        
        # Value estimate
        value = self.value_head(last_step_output)
        
        return action_logits, value, new_hidden_state
    
    def sample_action(
        self,
        state: torch.Tensor,
        hidden_state: Optional[Tuple] = None,
        temperature: float = 1.0,
        action_mask: Optional[torch.Tensor] = None,
        logit_bias: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple]]:
        """
        Sample action from policy distribution.

        Args:
            state: Current token [batch_size, 1]
            hidden_state: LSTM hidden state
            temperature: Sampling temperature (higher = more random)
            action_mask: Binary mask (1=allowed, 0=forbidden)
            logit_bias: Optional additive bias to logits (e.g. for operator preference)

        Returns:
            action: Sampled action index [batch_size]
            log_prob: Log probability of action [batch_size]
            value: State value estimate [batch_size, 1]
            new_hidden_state: Updated hidden state
        """
        # Forward pass
        logits, value, new_hidden_state = self.forward(state, hidden_state)

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply logit bias (cross-channel operator preference)
        if logit_bias is not None:
            logit_bias = logit_bias.to(logits.device)
            if logit_bias.dim() == 1:
                logit_bias = logit_bias.unsqueeze(0)
            logits = logits + logit_bias

        if action_mask is not None:
            if action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)
            action_mask = action_mask.to(logits.device)
            if action_mask.shape != logits.shape:
                raise ValueError(f"Action mask shape {action_mask.shape} does not match logits {logits.shape}")
            # Convert to boolean mask (1.0 = allowed, 0.0 = forbidden)
            action_mask_bool = action_mask.bool()
            masked_logits = logits.masked_fill(~action_mask_bool, -1e9)
        else:
            masked_logits = logits
            
        # Create distribution
        dist = torch.distributions.Categorical(logits=masked_logits)
        
        # Sample
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        if action_mask is not None:
            valid = action_mask_bool.gather(1, action.unsqueeze(1)).squeeze(1)
            if not torch.all(valid):
                raise RuntimeError("Sampled invalid action under mask")
        
        return action, log_prob, value, new_hidden_state
    
    def evaluate_actions(
        self,
        token_sequence: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Args:
            token_sequence: Full token sequences [batch_size, seq_len]
            actions: Actions taken at each step [batch_size, seq_len]
            
        Returns:
            log_probs: Log probabilities of actions [batch_size, seq_len]
            values: State value estimates [batch_size, seq_len]
            entropy: Entropy of distribution [batch_size, seq_len]
        """
        # Note: For LSTM, we usually process the entire sequence at once.
        # token_sequence should be the inputs (t=0..T-1)
        # actions should be the targets (t=1..T)? 
        # Or usually in PPO we align states and actions.
        # If token_sequence is [START, A, B], actions are [A, B, END].
        
        # We process the full sequence through LSTM
        # logits: [batch_size, seq_len, vocab_size]
        # values: [batch_size, seq_len, 1]
        
        # We need to handle hidden states if sequences are long, but here we assume full sequence fits.
        # Or we might be given just one step?
        # Usually for PPO with RNN, we unroll the whole sequence.
        
        embedded = self.embedding(token_sequence)
        
        if self.model_type == "lstm":
            output, _ = self.controller(embedded)
        else:
            raise NotImplementedError
            
        # [batch_size, seq_len, hidden_size]
        logits = self.action_head(output)
        values = self.value_head(output)
        
        dist = torch.distributions.Categorical(logits=logits)
        
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_probs, values.squeeze(-1), entropy
