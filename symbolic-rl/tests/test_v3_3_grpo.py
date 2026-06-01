"""Unit tests for v3.3 Section 2 — GRPO + Pareto group-relative advantage."""

import math
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.rl.formula_utils import (
    rpn_depth, fast_non_dominated_sort, crowding_distance,
)
from src.rl.grpo_trainer import GRPOTrainer


# --------------------------------------------------------------------------
# Depth (acceptance: the two specced cases + nesting)
# --------------------------------------------------------------------------

def test_rpn_depth():
    assert rpn_depth("I_R edge_x pool_center".split()) == 3
    assert rpn_depth("I_R I_G add pool_center".split()) == 3
    assert rpn_depth("I_R".split()) == 1                      # bare terminal
    assert rpn_depth([]) == 0
    # nested: ((I_R edge_x) (I_G edge_y) add) global_avg_pool
    #  I_R=1, edge_x=2, I_G=1, edge_y=2, add=max(2,2)+1=3, pool=4
    assert rpn_depth("I_R edge_x I_G edge_y add global_avg_pool".split()) == 4


# --------------------------------------------------------------------------
# Non-dominated sort (acceptance: 5 hand-checked (acc, len, depth) triples)
# --------------------------------------------------------------------------

def test_fast_non_dominated_sort_handchecked():
    # objectives = (accuracy↑, length↓, depth↓)
    A = (0.50, 10, 3)
    B = (0.50, 12, 4)   # A dominates B (equal acc, A shorter & shallower) -> front 1
    C = (0.55, 20, 6)   # highest acc -> front 0
    D = (0.40, 5, 2)    # shortest/shallowest -> front 0
    E = (0.45, 8, 3)    # non-dominated -> front 0
    objs = [A, B, C, D, E]
    directions = ['max', 'min', 'min']
    ranks, fronts = fast_non_dominated_sort(objs, directions)
    # A,C,D,E on the Pareto front (rank 0); only B is dominated (rank 1)
    assert ranks == [0, 1, 0, 0, 0], ranks
    assert set(fronts[0]) == {0, 2, 3, 4}
    assert fronts[1] == [1]


def test_crowding_distance_boundaries_inf():
    objs = [(0.4, 5, 2), (0.45, 8, 3), (0.5, 10, 3), (0.55, 20, 6)]
    front = [0, 1, 2, 3]
    cd = crowding_distance(objs, front, ['max', 'min', 'min'])
    # extreme points along sorted axes get +inf
    assert cd[0] == math.inf and cd[3] == math.inf
    # interior points are finite
    assert math.isfinite(cd[1]) and math.isfinite(cd[2])


# --------------------------------------------------------------------------
# Parsimony tie-break ("强行卡死"): equal accuracy -> shorter/shallower wins
# --------------------------------------------------------------------------

def _make_trainer(env=None, policy=None):
    return GRPOTrainer(policy=policy or nn.Linear(1, 1), env=env,
                       device='cpu', group_size=4, n_epochs=1, batch_size=4)


def test_parsimony_tiebreak():
    tr = _make_trainer()
    # two formulas, identical accuracy; one shorter+shallower
    acc =   [0.50, 0.50]
    length = [5,    15]
    depth =  [2,    6]
    adv, ranks = tr.compute_group_advantage(acc, length, depth)
    # both on the Pareto front, but the shorter/shallower one gets higher advantage
    assert adv[0] > adv[1], (adv, ranks)


def test_advantage_is_normalized():
    tr = _make_trainer()
    acc = [0.6, 0.5, 0.4, 0.3]
    length = [8, 10, 12, 6]
    depth = [3, 4, 5, 2]
    adv, _ = tr.compute_group_advantage(acc, length, depth)
    assert abs(float(adv.mean())) < 1e-5          # zero-mean
    assert abs(float(adv.std(unbiased=False)) - 1.0) < 0.2  # ~unit std


# --------------------------------------------------------------------------
# Mock end-to-end update() — verifies the critic-free optimization loop runs
# --------------------------------------------------------------------------

class _MockVocab:
    def __init__(self):
        self.tokens = ['START', 'END', 'PAD', 'I_R', 'edge_x', 'pool_center']
        self.token_to_idx = {t: i for i, t in enumerate(self.tokens)}
        self.idx_to_token = {i: t for i, t in enumerate(self.tokens)}
    def encode(self, t): return self.token_to_idx[t]
    def decode(self, i): return self.idx_to_token[int(i)]
    def __len__(self): return len(self.tokens)


class _MockEnv:
    """Generates a fixed-length 3-token formula 'I_R edge_x pool_center'."""
    def __init__(self):
        self.vocabulary = _MockVocab()
        self._n = 0
        self._seq = ['I_R', 'edge_x', 'pool_center']
        self._calls = 0
    def reset(self):
        self._n = 0
        return None, {}
    def get_action_mask(self):
        return torch.ones(len(self.vocabulary))
    def step(self, action):
        self._n += 1
        done = self._n >= 3
        info = {}
        reward = 0.0
        if done:
            # vary accuracy per rollout so the group has spread
            self._calls += 1
            acc = 0.3 + 0.1 * (self._calls % 4)
            info = {'accuracy': acc, 'formula': 'I_R edge_x pool_center', 'valid': True}
            reward = acc
        return None, reward, done, False, info


class _MockPolicy(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 8)
        self.action_head = nn.Linear(8, vocab_size)
        self.value_head = nn.Linear(8, 1)
        self.vocab_size = vocab_size
    def sample_action(self, curr, hidden, action_mask=None, logit_bias=None):
        h = self.emb(curr).mean(dim=1)               # [1, 8]
        logits = self.action_head(h)                 # [1, V]
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()                       # [1]
        log_prob = dist.log_prob(action)             # [1]
        value = self.value_head(h).squeeze(-1)
        return action, log_prob, value, hidden
    def evaluate_actions(self, states, actions):
        h = self.emb(states)                         # [B, T, 8]
        logits = self.action_head(h)                 # [B, T, V]
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)           # [B, T]
        entropy = dist.entropy()                     # [B, T]
        values = self.value_head(h).squeeze(-1)      # [B, T]
        return log_probs, values, entropy


def test_grpo_update_smoke():
    torch.manual_seed(0)
    env = _MockEnv()
    policy = _MockPolicy(len(env.vocabulary))
    tr = GRPOTrainer(policy=policy, env=env, device='cpu',
                     group_size=8, n_epochs=2, batch_size=4, learning_rate=1e-2)
    before = [p.clone() for p in policy.parameters()]
    metrics = tr.update(n_episodes=8)
    assert 'loss' in metrics and 'mean_length' in metrics
    assert metrics['value_loss'] == 0.0              # critic-free
    assert metrics['mean_length'] == 3               # fixed 3-token formula
    assert metrics['pareto_front_size'] >= 1
    assert len(tr.mean_length_history) == 1
    # at least one parameter changed (optimization actually ran)
    after = list(policy.parameters())
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f"ok: {name}")
    print('All v3.3 Section 2 GRPO tests passed.')
