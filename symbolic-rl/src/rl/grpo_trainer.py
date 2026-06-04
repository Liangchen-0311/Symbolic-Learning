"""
GRPO (Group Relative Policy Optimization) trainer with Pareto group-relative advantage
(v3.3 Section 2).

Why GRPO + Pareto (recorded per Section 2D):
  - GRPO DROPS THE CRITIC. On symbolic search the critic has a near-impossible job: it
    must predict the value of half-finished, non-executable formulas in a highly
    discontinuous reward landscape. Group-relative ranking sidesteps this entirely —
    advantage comes from how a formula ranks *within its sampled group*, not from a
    learned value function.
  - PARETO DOMINANCE over (accuracy↑, length↓, depth↓) prevents BLOAT without a
    hand-tuned weight λ: a 51%-accuracy / length-50 formula does NOT dominate a
    50%-accuracy / length-10 formula, so the search cannot trade length for tiny
    accuracy gains. A parsimony tie-break ("强行卡死") then strictly prefers the
    shorter/shallower formula whenever accuracies are within ``acc_tol``.

Outward interface matches ``PPOTrainer`` (``__init__(policy, env, ...)``, ``update``,
``train``, ``save_checkpoint``, ``set_binary_op_bias``) so the entrypoint can swap them.
The ``policy.value_head`` is left intact for backward compat but is NOT used or trained.
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List

from src.rl.formula_utils import (
    rpn_depth, fast_non_dominated_sort, crowding_distance,
)


class GRPOTrainer:
    """Critic-free, group-relative, Pareto-ranked policy optimization."""

    def __init__(
        self,
        policy,
        env,
        learning_rate: float = 3e-4,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: str = "cuda",
        # GRPO / Pareto
        group_size: int = 16,
        acc_tol: float = 0.003,
        crowding_weight: float = 0.1,
        # Section 2E: classifier-dependent parsimony strength. The orchestrator sets this
        # from classifier.type — linear → strict (1e-3, anti-bloat is free), histgb →
        # relaxed (2e-4 or 0, keep useful long non-linear formulas).
        lambda_len: float = 1.0e-3,
        # Entropy schedule (same semantics as PPOTrainer)
        entropy_coef_start: float = None,
        entropy_coef_end: float = None,
        entropy_decay_fraction: float = 0.5,
        lr_warmup_iterations: int = 0,
        total_iterations: int = 1000,
        # accepted-but-unused (kept so PPO kwargs can be forwarded verbatim)
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        value_coef: float = 0.5,
    ):
        self.policy = policy
        self.env = env
        self.device = device

        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # GRPO / Pareto
        self.group_size = group_size
        self.acc_tol = acc_tol
        self.crowding_weight = crowding_weight
        self.lambda_len = lambda_len   # Section 2E: classifier-dependent (see __init__)

        # Entropy schedule + LR warmup (mirror PPOTrainer)
        self.entropy_coef_start = entropy_coef_start or entropy_coef
        self.entropy_coef_end = entropy_coef_end
        self.entropy_decay_fraction = entropy_decay_fraction
        self.total_iterations = total_iterations
        self.lr_warmup_iterations = lr_warmup_iterations

        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.iteration_count = 0

        self.episode_rewards = []
        self.best_reward = float('-inf')
        self.best_program = "None"

        # Bloat tracking (acceptance: mean length must not grow monotonically)
        self.mean_length_history: List[float] = []
        self.mean_depth_history: List[float] = []

        # Binary operator bias (same as PPO)
        self.binary_op_bias = 0.0
        self._binary_op_indices = None

    # -- shared helpers (parity with PPOTrainer) ---------------------------
    def set_binary_op_bias(self, bias: float, vocabulary):
        self.binary_op_bias = bias
        if bias > 0 and vocabulary is not None:
            self._binary_op_indices = []
            for name in ['subtract', 'multiply']:
                if name in vocabulary.token_to_idx:
                    self._binary_op_indices.append(vocabulary.token_to_idx[name])

    def _apply_binary_bias(self, action_mask):
        if self.binary_op_bias <= 0 or self._binary_op_indices is None:
            return None
        bias = torch.zeros(action_mask.shape[-1], device=action_mask.device)
        for idx in self._binary_op_indices:
            if action_mask.dim() == 1:
                if action_mask[idx] > 0:
                    bias[idx] = self.binary_op_bias
            else:
                bias[idx] = self.binary_op_bias
        return bias

    def _update_schedule(self):
        t = self.iteration_count
        if self.entropy_coef_end is not None:
            decay_iters = int(self.total_iterations * self.entropy_decay_fraction)
            if decay_iters > 0 and t < decay_iters:
                frac = t / decay_iters
                self.entropy_coef = (
                    self.entropy_coef_start * (1 - frac) + self.entropy_coef_end * frac
                )
            elif decay_iters > 0:
                self.entropy_coef = self.entropy_coef_end
        if self.lr_warmup_iterations > 0 and t < self.lr_warmup_iterations:
            warmup_factor = (t + 1) / self.lr_warmup_iterations
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.learning_rate * warmup_factor

    # -- group rollout -----------------------------------------------------
    def collect_group(self, group_size: int) -> Dict[str, List]:
        """Sample a group of ``group_size`` complete formulas from the current policy.

        Per formula returns its trajectory (states/actions/log_probs) plus the three
        Pareto objectives: accuracy (↑), length (↓), depth (↓).
        """
        g = {
            "states": [], "actions": [], "log_probs": [],
            "rewards": [], "accuracy": [], "length": [], "depth": [],
            "formula": [],
        }
        self.policy.eval()
        for _ in range(group_size):
            self.env.reset()
            ep_states, ep_actions, ep_logprobs = [], [], []
            terminated = truncated = False
            hidden = None
            curr = torch.tensor([[self.env.vocabulary.encode('START')]], device=self.device)
            info = {}
            reward = 0.0
            while not (terminated or truncated):
                ep_states.append(curr)
                with torch.no_grad():
                    action_mask = self.env.get_action_mask().to(self.device)
                    logit_bias = self._apply_binary_bias(action_mask)
                    action, log_prob, _value, hidden = self.policy.sample_action(
                        curr, hidden, action_mask=action_mask, logit_bias=logit_bias)
                action_item = action.item()
                _obs, reward, terminated, truncated, info = self.env.step(action_item)
                ep_actions.append(action)
                ep_logprobs.append(log_prob)
                curr = action.unsqueeze(1)

            # Pareto objectives
            accuracy = float(info.get('accuracy', 0.0)) if isinstance(info, dict) else 0.0
            formula_str = info.get('formula', '') if isinstance(info, dict) else ''
            if formula_str:
                decoded = formula_str.split()
            else:
                decoded = [self.env.vocabulary.decode(a.item()) for a in ep_actions]
                decoded = [t for t in decoded if t not in ('START', 'END', 'PAD')]
            length = len(decoded)
            depth = rpn_depth(decoded)

            g["states"].append(torch.cat(ep_states, dim=1))
            g["actions"].append(torch.stack(ep_actions, dim=1))
            g["log_probs"].append(torch.stack(ep_logprobs, dim=1))
            g["rewards"].append(reward)
            g["accuracy"].append(accuracy)
            g["length"].append(length)
            g["depth"].append(depth)
            g["formula"].append(formula_str)

            if reward > self.best_reward:
                self.best_reward = reward
                self.best_program = formula_str or "N/A"
        return g

    # -- Pareto group-relative advantage -----------------------------------
    def compute_group_advantage(self, accuracy, length, depth):
        """Pareto non-dominated rank + crowding distance -> normalized scalar advantage
        per formula. Implements the parsimony tie-break (Section 2.7 / "强行卡死")."""
        n = len(accuracy)
        objectives = list(zip(accuracy, length, depth))
        directions = ['max', 'min', 'min']           # accuracy↑, length↓, depth↓
        ranks, fronts = fast_non_dominated_sort(objectives, directions)

        # crowding distance per individual (computed within its front)
        crowd = [0.0] * n
        for front in fronts:
            cd = crowding_distance(objectives, front, directions)
            for i, d in cd.items():
                crowd[i] = d
        # normalize crowding to [0,1] (inf -> 1.0; preserves boundary preference)
        finite = [c for c in crowd if c != float('inf')]
        cmax = max(finite) if finite else 1.0
        norm_crowd = [1.0 if c == float('inf') else (c / cmax if cmax > 0 else 0.0)
                      for c in crowd]

        raw = torch.empty(n, dtype=torch.float32)
        for i in range(n):
            raw[i] = -float(ranks[i]) + self.crowding_weight * norm_crowd[i]
        # parsimony tie-break: within a front, shorter/shallower gets strictly higher
        # raw_score (also enforces the acc_tol "强行卡死" preference lexicographically).
        # Strength is classifier-dependent (Section 2E): self.lambda_len.
        for i in range(n):
            raw[i] -= self.lambda_len * length[i] + self.lambda_len * depth[i]

        adv = (raw - raw.mean()) / (raw.std() + 1e-8)
        return adv, ranks

    # -- update ------------------------------------------------------------
    def update(self, n_episodes: int = None) -> Dict[str, float]:
        """One GRPO update: sample a group, Pareto-rank it, assign each formula's scalar
        advantage to all its tokens, and run the clipped surrogate (NO value loss)."""
        self._update_schedule()
        self.iteration_count += 1
        G = n_episodes or self.group_size

        group = self.collect_group(G)
        adv, ranks = self.compute_group_advantage(
            group["accuracy"], group["length"], group["depth"])
        adv = adv.to(self.device)

        pad_id = self.env.vocabulary.encode('PAD')
        padded_states = nn.utils.rnn.pad_sequence(
            [s.squeeze(0) for s in group["states"]], batch_first=True, padding_value=pad_id)
        padded_actions = nn.utils.rnn.pad_sequence(
            [a.squeeze(0) for a in group["actions"]], batch_first=True, padding_value=pad_id)
        padded_old_log_probs = nn.utils.rnn.pad_sequence(
            [p.squeeze(0) for p in group["log_probs"]], batch_first=True, padding_value=0)
        mask = (padded_states != pad_id).float()

        # trajectory-level credit assignment: broadcast each formula's scalar advantage
        # across all of its tokens (standard GRPO).
        padded_adv = adv.unsqueeze(1).expand_as(mask)

        self.policy.train()
        n_samples = padded_states.size(0)
        total_loss = total_policy = total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(n_samples)
            for start in range(0, n_samples, self.batch_size):
                bidx = perm[start:start + self.batch_size]
                b_states = padded_states[bidx]
                b_actions = padded_actions[bidx]
                b_adv = padded_adv[bidx]
                b_old_lp = padded_old_log_probs[bidx]
                b_mask = mask[bidx]

                log_probs, _values, entropy = self.policy.evaluate_actions(b_states, b_actions)

                ratio = torch.exp(log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2)
                entropy_loss = -entropy

                policy_loss = (policy_loss * b_mask).sum() / (b_mask.sum() + 1e-8)
                entropy_loss = (entropy_loss * b_mask).sum() / (b_mask.sum() + 1e-8)
                # NO value loss term (critic-free).
                loss = policy_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss += loss.item()
                total_policy += policy_loss.item()
                total_entropy += entropy_loss.item()
                n_updates += 1

        mean_len = sum(group["length"]) / len(group["length"])
        mean_depth = sum(group["depth"]) / len(group["depth"])
        self.mean_length_history.append(mean_len)
        self.mean_depth_history.append(mean_depth)
        self.episode_rewards.extend(group["rewards"])

        n_updates = max(n_updates, 1)
        return {
            "loss": total_loss / n_updates,
            "policy_loss": total_policy / n_updates,
            "value_loss": 0.0,                       # critic-free
            "entropy": -total_entropy / n_updates,
            "entropy_coef": self.entropy_coef,
            "avg_reward": sum(group["rewards"]) / len(group["rewards"]),
            "avg_accuracy": sum(group["accuracy"]) / len(group["accuracy"]),
            "mean_length": mean_len,
            "mean_depth": mean_depth,
            "pareto_front_size": int(sum(1 for r in ranks if r == 0)),
        }

    def save_checkpoint(self, path: str, name: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(path, name))

    def train(self, n_iterations: int, episodes_per_iteration: int,
              save_dir: str = "./outputs/checkpoints") -> None:
        for iteration in range(n_iterations):
            t0 = time.time()
            print(f"\n=== [GRPO] Iteration {iteration + 1}/{n_iterations} ===")
            metrics = self.update(n_episodes=episodes_per_iteration)
            print(f"Duration: {time.time() - t0:.2f}s")
            print(f"Avg Reward: {metrics['avg_reward']:.4f} | Avg Acc: {metrics['avg_accuracy']:.4f}")
            print(f"Mean length: {metrics['mean_length']:.2f} | Mean depth: {metrics['mean_depth']:.2f} "
                  f"| Pareto front: {metrics['pareto_front_size']}")
            print(f"Loss: {metrics['loss']:.4f} (Policy: {metrics['policy_loss']:.4f}, "
                  f"Entropy: {metrics['entropy']:.4f})")
            print(f"Best Reward So Far: {self.best_reward:.4f} | Best: {self.best_program}")
            if (iteration + 1) % 10 == 0:
                self.save_checkpoint(save_dir, f"grpo_checkpoint_iter_{iteration+1}.pth")
