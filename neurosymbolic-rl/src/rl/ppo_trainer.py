"""
PPO (Proximal Policy Optimization) Trainer for symbolic expression search.

This implements the main RL training loop:
1. Collect trajectories using current policy
2. Compute advantages and returns
3. Update policy using PPO objective
4. Update value function

References:
- PPO Paper: https://arxiv.org/abs/1707.06347
- Stable implementation pattern
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict
from tqdm import tqdm

from src.models.policy_agent import PolicyAgent
from src.rl.environment import SymbolicExpressionEnv
from src.symbolic.tensor_operators import TENSOR_OPERATORS


class PPOTrainer:
    """
    PPO trainer for policy agent.
    
    Training Loop:
    1. Collect N episodes using current policy
    2. Compute returns and advantages (GAE)
    3. Perform K epochs of minibatch updates
    4. Log metrics and save checkpoints
    """
    
    def __init__(
        self,
        policy: PolicyAgent,
        env: SymbolicExpressionEnv,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cuda",
        # Entropy schedule
        entropy_coef_start: float = None,
        entropy_coef_end: float = None,
        entropy_decay_fraction: float = 0.5,
        # LR warmup
        lr_warmup_iterations: int = 0,
        # Total iterations (needed for scheduling)
        total_iterations: int = 1000,
    ):
        """
        Args:
            policy: Policy agent (LSTM/Transformer controller)
            env: Symbolic expression environment
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient (or initial if schedule is set)
            max_grad_norm: Gradient clipping threshold
            n_epochs: Number of PPO update epochs per iteration
            batch_size: Minibatch size for updates
            device: Device for training
            entropy_coef_start: Start entropy coef (if None, uses entropy_coef)
            entropy_coef_end: End entropy coef (if None, no schedule)
            entropy_decay_fraction: Fraction of training over which to decay
            lr_warmup_iterations: Number of iterations for linear LR warmup
            total_iterations: Total number of training iterations
        """
        self.policy = policy
        self.env = env
        self.device = device

        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Entropy schedule
        self.entropy_coef_start = entropy_coef_start or entropy_coef
        self.entropy_coef_end = entropy_coef_end
        self.entropy_decay_fraction = entropy_decay_fraction
        self.total_iterations = total_iterations

        # LR warmup
        self.lr_warmup_iterations = lr_warmup_iterations

        # Optimizer
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

        # Iteration counter (for scheduling)
        self.iteration_count = 0

        # Metrics
        self.episode_rewards = []
        self.best_reward = float('-inf')
        self.best_program = "None"

        # Binary operator bias
        self.binary_op_bias = 0.0
        self._binary_op_indices = None
    
    def set_binary_op_bias(self, bias: float, vocabulary):
        """Set logit bias for cross-channel binary operators (subtract, multiply)."""
        self.binary_op_bias = bias
        if bias > 0 and vocabulary is not None:
            self._binary_op_indices = []
            for name in ['subtract', 'multiply']:
                if name in vocabulary.token_to_idx:
                    self._binary_op_indices.append(vocabulary.token_to_idx[name])

    def _apply_binary_bias(self, action_mask: 'torch.Tensor') -> 'torch.Tensor':
        """Add positive logit bias to the action mask for binary operators.

        This doesn't change the mask but returns a bias tensor to be added
        to logits during sampling.
        """
        if self.binary_op_bias <= 0 or self._binary_op_indices is None:
            return None
        bias = torch.zeros(action_mask.shape[-1], device=action_mask.device)
        for idx in self._binary_op_indices:
            # Only bias if the action is actually allowed
            if action_mask.dim() == 1:
                if action_mask[idx] > 0:
                    bias[idx] = self.binary_op_bias
            else:
                bias[idx] = self.binary_op_bias  # mask will block forbidden anyway
        return bias

    def _update_schedule(self):
        """Update entropy coef and LR based on current iteration."""
        t = self.iteration_count

        # Entropy schedule: linear decay over first entropy_decay_fraction of training
        if self.entropy_coef_end is not None:
            decay_iters = int(self.total_iterations * self.entropy_decay_fraction)
            if decay_iters > 0 and t < decay_iters:
                frac = t / decay_iters
                self.entropy_coef = (
                    self.entropy_coef_start * (1 - frac)
                    + self.entropy_coef_end * frac
                )
            elif decay_iters > 0:
                self.entropy_coef = self.entropy_coef_end

        # LR warmup: linear from 0 to base LR
        if self.lr_warmup_iterations > 0 and t < self.lr_warmup_iterations:
            warmup_factor = (t + 1) / self.lr_warmup_iterations
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.learning_rate * warmup_factor

    def collect_trajectories(
        self,
        n_episodes: int
    ) -> Dict[str, List]:
        """
        Collect trajectories using current policy.
        
        Args:
            n_episodes: Number of episodes to collect
        
        Returns:
            Dictionary containing:
            - states: List of state sequences
            - actions: List of action sequences
            - log_probs: List of log probabilities
            - rewards: List of episode rewards
            - values: List of value estimates
        """
        trajectories = {
            "states": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "values": []
        }
        
        self.policy.eval()
        
        for _ in range(n_episodes):
            state, _ = self.env.reset()
            episode_states = []
            episode_actions = []
            episode_log_probs = []
            episode_values = []
            
            terminated = False
            truncated = False
            hidden_state = None
            
            # Initial input: START token
            # We need to construct the input for the policy.
            # If policy takes [batch, 1], we need to pass tensor.
            curr_token = torch.tensor([[self.env.vocabulary.encode('START')]], device=self.device)
            
            while not (terminated or truncated):
                # Store state (token)
                episode_states.append(curr_token)
                
                # Sample action
                with torch.no_grad():
                    action_mask = self.env.get_action_mask().to(self.device)
                    logit_bias = self._apply_binary_bias(action_mask)
                    action, log_prob, value, hidden_state = self.policy.sample_action(
                        curr_token, hidden_state, action_mask=action_mask,
                        logit_bias=logit_bias
                    )
                
                # Step environment
                action_item = action.item()
                obs, reward, terminated, truncated, info = self.env.step(action_item)
                
                # Store data
                episode_actions.append(action)
                episode_log_probs.append(log_prob)
                episode_values.append(value)
                
                # Next input is the action we just took
                curr_token = action.unsqueeze(1) # [1, 1]
            
            # Store episode reward
            trajectories["rewards"].append(reward)
            
            # Track best program
            if reward > self.best_reward:
                print(f"DEBUG: New best reward {reward}. Info keys: {info.keys()}")
                self.best_reward = reward
                if "program_str" in info:
                    self.best_program = info["program_str"]
                else:
                    self.best_program = f"N/A (Valid: {info.get('valid', 'Unknown')})"
            
            # Store trajectory tensors
            # Stack lists to tensors
            trajectories["states"].append(torch.cat(episode_states, dim=1)) # [1, seq_len]
            trajectories["actions"].append(torch.stack(episode_actions, dim=1)) # [1, seq_len]
            trajectories["log_probs"].append(torch.stack(episode_log_probs, dim=1)) # [1, seq_len]
            trajectories["values"].append(torch.stack(episode_values, dim=1)) # [1, seq_len]
            
        return trajectories

    def compute_gae(
        self,
        rewards: List[float],
        values: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute Generalized Advantage Estimation (GAE).
        """
        advantages = []
        returns = []
        
        for r, v_seq in zip(rewards, values):
            # v_seq: [1, seq_len]
            T = v_seq.size(1)
            advs = torch.zeros_like(v_seq)
            rets = torch.zeros_like(v_seq)
            
            final_reward = r
            last_gae_lam = 0
            
            for t in reversed(range(T)):
                if t == T - 1:
                    next_val = 0
                    reward_t = final_reward
                else:
                    next_val = v_seq[0, t+1].item()
                    reward_t = 0
                
                current_val = v_seq[0, t].item()
                
                delta = reward_t + self.gamma * next_val - current_val
                last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam
                
                advs[0, t] = last_gae_lam
                rets[0, t] = current_val + last_gae_lam
            
            advantages.append(advs)
            returns.append(rets)
            
        return advantages, returns

    def update(self, n_episodes: int = 100) -> Dict[str, float]:
        """
        Perform one PPO update cycle.
        """
        # Update schedules (entropy decay, LR warmup)
        self._update_schedule()
        self.iteration_count += 1

        # 1. Collect
        trajectories = self.collect_trajectories(n_episodes)
        
        # 2. Compute GAE
        advantages, returns = self.compute_gae(trajectories["rewards"], trajectories["values"])
        
        # Prepare batch data
        padded_states = torch.nn.utils.rnn.pad_sequence([s.squeeze(0) for s in trajectories["states"]], batch_first=True, padding_value=self.env.vocabulary.encode('PAD'))
        padded_actions = torch.nn.utils.rnn.pad_sequence([a.squeeze(0) for a in trajectories["actions"]], batch_first=True, padding_value=self.env.vocabulary.encode('PAD'))
        padded_advantages = torch.nn.utils.rnn.pad_sequence([a.squeeze(0) for a in advantages], batch_first=True, padding_value=0).squeeze(-1)
        padded_returns = torch.nn.utils.rnn.pad_sequence([r.squeeze(0) for r in returns], batch_first=True, padding_value=0).squeeze(-1)
        padded_old_log_probs = torch.nn.utils.rnn.pad_sequence([p.squeeze(0) for p in trajectories["log_probs"]], batch_first=True, padding_value=0)
        
        mask = (padded_states != self.env.vocabulary.encode('PAD')).float()
        
        # Normalize advantages
        valid_advs = padded_advantages[mask.bool()]
        if valid_advs.numel() > 1:
            padded_advantages = (padded_advantages - valid_advs.mean()) / (valid_advs.std() + 1e-8)
        
        # 3. PPO Update
        self.policy.train()
        
        n_samples = padded_states.size(0)
        indices = torch.arange(n_samples)
        
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        for _ in range(self.n_epochs):
            indices = torch.randperm(n_samples)
            
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                batch_indices = indices[start_idx:end_idx]
                
                b_states = padded_states[batch_indices]
                b_actions = padded_actions[batch_indices]
                b_returns = padded_returns[batch_indices]
                b_advantages = padded_advantages[batch_indices]
                b_old_log_probs = padded_old_log_probs[batch_indices]
                b_mask = mask[batch_indices]
                
                log_probs, values, entropy = self.policy.evaluate_actions(b_states, b_actions)
                
                ratio = torch.exp(log_probs - b_old_log_probs)
                
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_advantages
                
                policy_loss = -torch.min(surr1, surr2)
                value_loss = 0.5 * (values - b_returns).pow(2)
                entropy_loss = -entropy
                
                policy_loss = (policy_loss * b_mask).sum() / (b_mask.sum() + 1e-8)
                value_loss = (value_loss * b_mask).sum() / (b_mask.sum() + 1e-8)
                entropy_loss = (entropy_loss * b_mask).sum() / (b_mask.sum() + 1e-8)
                
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
        
        num_updates = self.n_epochs * (n_samples // self.batch_size + 1)
        
        metrics = {
            "loss": total_loss / num_updates,
            "policy_loss": total_policy_loss / num_updates,
            "value_loss": total_value_loss / num_updates,
            "entropy": -total_entropy / num_updates,
            "entropy_coef": self.entropy_coef,
            "avg_reward": sum(trajectories["rewards"]) / len(trajectories["rewards"])
        }
        
        return metrics

    def save_checkpoint(self, path: str, name: str) -> None:
        """Save policy checkpoint."""
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, name))

    def train(
        self,
        n_iterations: int,
        episodes_per_iteration: int,
        save_dir: str = "./outputs/checkpoints"
    ) -> None:
        """
        Main training loop.
        
        Args:
            n_iterations: Number of training iterations
            episodes_per_iteration: Episodes to collect per iteration
            save_dir: Directory to save checkpoints
        """
        import time
        
        for iteration in range(n_iterations):
            start_time = time.time()
            print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")
            
            # Perform update
            metrics = self.update(n_episodes=episodes_per_iteration)
            
            # Log metrics
            duration = time.time() - start_time
            print(f"Duration: {duration:.2f}s")
            print(f"Average Reward: {metrics['avg_reward']:.4f}")
            print(f"Loss: {metrics['loss']:.4f} (Policy: {metrics['policy_loss']:.4f}, Value: {metrics['value_loss']:.4f}, Entropy: {metrics['entropy']:.4f})")
            print(f"Best Reward So Far: {self.best_reward:.4f}")
            print(f"Best Program So Far: {self.best_program}")
            
            # Save best model
            if metrics['avg_reward'] > self.best_reward:
                self.best_reward = metrics['avg_reward']
                self.save_checkpoint(save_dir, "best_model.pth")
                print(f"New best reward! Saved checkpoint.")
            
            # Periodic save
            if (iteration + 1) % 10 == 0:
                 self.save_checkpoint(save_dir, f"checkpoint_iter_{iteration+1}.pth")
