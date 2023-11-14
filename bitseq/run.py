import argparse
import numpy as np
import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple
from scipy.stats import spearmanr

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset
from torch.distributions.categorical import Categorical

from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler
from tensordict import TensorDict

parser = argparse.ArgumentParser()

parser.add_argument("--n", default=120, type=int)
parser.add_argument("--k", default=4, type=int)
parser.add_argument("--M_size", default=60, type=int)
parser.add_argument("--mode_threshold", default=30, type=int)
parser.add_argument("--reward_exponent", default=2.0, type=float)
parser.add_argument("--seed", default=0, type=int)

parser.add_argument("--device", default='cuda', type=str)
parser.add_argument("--num_iterations", default=50000, type=int)
parser.add_argument("--rand_action_prob", default=0.001, type=float)
parser.add_argument("--learning_rate", default=0.001, type=float)
parser.add_argument("--dropout", default=0.1, type=float)
parser.add_argument("--batch_size", default=16, type=int)
parser.add_argument("--print_every", default=50, type=int)
parser.add_argument("--print_modes", default=False, action='store_true')

parser.add_argument("--objective", default='tb', type=str)
parser.add_argument("--z_learning_rate", default=0.001, type=float)
parser.add_argument("--subtb_lambda", default=1.9, type=float)
parser.add_argument("--leaf_coeff", default=5.0, type=float)
parser.add_argument("--update_target_every", default=5, type=int)
parser.add_argument("--corr_num_rounds", default=10, type=int)

# SoftQL params
parser.add_argument("--start_learning", default=50, type=int)
parser.add_argument("--softql_loss", default='Huber', type=str)

# Replay buffer parameters
parser.add_argument("--rb_size", default=100_000, type=int)
parser.add_argument("--rb_batch_size", default=256, type=int)
parser.add_argument("--per_alpha", default=0.9, type=float)
parser.add_argument("--per_beta", default=0.1, type=float)
parser.add_argument("--anneal_per_beta", default=False, action='store_true')

# Munchausen DQN parameters
parser.add_argument("--m_alpha", default=0.0, type=float)
parser.add_argument("--entropy_coeff", default=1.0, type=float)
parser.add_argument("--m_l0", default=-25.0, type=float)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.2, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, seq_len: int, dropout: float = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 2)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken + seq_len + 1)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        src = self.embedding(src) 
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.linear(output)
        return output


def construct_M(n, b, H, M_size, seed=0):
    np.random.seed(seed) 
    M = []
    for i in range(M_size):
        M.append("".join([np.random.choice(H) for _ in range(n // b)]))
        assert len(M[-1]) == n
    return M

def distance(s1, s2):
    assert len(s1) == len(s2)
    return sum([int(s1[i] != s2[i]) for i in range(len(s1))])

def M_distance(s, M):
    return min([distance(s, ms) for ms in M])

def construct_test_set(M, seed=0):
    np.random.seed(seed) 
    test_set = []
    for s in M:
        test_set.append(s)
        for cnt in range(1, len(s)):
            new_s = list(s)
            subset = np.random.choice(list(range(len(s))), size=cnt, replace=False)
            for i in subset:
                new_s[i] = "0" if s[i] == "1" else "1"
            test_set.append("".join(new_s))
            assert len(test_set[-1]) == len(s)
            assert distance(test_set[-1], s) == cnt
    return test_set

def log_reward(s, M):
    return -M_distance(s, M)

def reward(s, M):
    return np.exp(log_reward(s, M))

def token_seq_to_str(seq, k):
    return "".join([bin(int(v))[2:].zfill(k) for v in seq])

def batch_rewards(batch, M, k):
    batch_np = batch.cpu().numpy()
    rewards = [reward(token_seq_to_str(batch_np[i], k), M) for i in range(batch_np.shape[0])]
    return torch.tensor(rewards)

def batch_log_rewards(batch, M, k):
    batch_np = batch.cpu().numpy()
    log_rewards = [log_reward(token_seq_to_str(batch_np[i], k), M) for i in range(batch_np.shape[0])]
    return torch.tensor(log_rewards)

def process_logits(all_logits, pos_mask, args):
    # Model predicts positional logits p_i and word logits for each position w_ij.
    # The logits used to sample pairs of positions and word (i, j) are computed as p_i + w_ij.
    pos_logits = all_logits[0, :, -(args.n // args.k + 1):] # [batch_size, n/k + 1]
    pos_logits[pos_mask] = float("-inf")
    word_logits = all_logits[:, :, :2**args.k] # [n/k + 1, batch_size, 2^k]
    sum_logits = torch.moveaxis(word_logits, 1, 0) + pos_logits[:, :, None] #[batch_size, n/k + 1, 2^k]
    sum_logits = sum_logits.reshape(pos_logits.shape[0], (args.n // args.k + 1) * (2 ** args.k)) #[batch_size, (n/k + 1) * 2^k]
    return pos_logits, word_logits, sum_logits

def sample_forward(sum_logits, sum_uniform, batch, args):
    # There is a bug in pytorch that allows to sample objects that has 0 probability (happens very rarely but still happens).
    # This loop basically resamples until everything is correct.
    while True:
        actions = Categorical(logits=sum_logits.clone()).sample()
        uniform_actions = Categorical(logits=sum_uniform).sample().to(args.device)
        uniform_mask = torch.rand(args.batch_size) < args.rand_action_prob
        actions[uniform_mask] = uniform_actions[uniform_mask]
        positions = actions // (2 ** args.k)
        if (batch[range(args.batch_size), positions] == 2 ** args.k).sum() == args.batch_size:
            break
    assert positions.min() >= 1
    assert positions.max() <= args.n // args.k
    words = actions % (2 ** args.k)
    return actions, positions, words


def TB_train_step(model, log_Z, optimizer, Z_optimizer, M, args):
    # This code is pretty simple because all trajectories in our graph have the same length.
    model.train()
    optimizer.zero_grad()
    Z_optimizer.zero_grad()

    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    batch = torch.tensor([[2 ** args.k + 1] + ([2 ** args.k] * (args.n // args.k)) for i in range(args.batch_size)]).to(args.device)
    p_forward_sum = torch.zeros(args.batch_size).to(args.device)
    p_backward_sum = torch.zeros(args.batch_size).to(args.device)

    for i in range(args.n // args.k):
        pos_mask = batch != 2 ** args.k
        all_logits = model(batch.T)
        pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)

        with torch.no_grad():
            _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

            actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args)

            batch_cl = batch.clone()
            batch_cl[range(args.batch_size), positions] = words
            batch = batch_cl
 
        p_forward_sum += sum_logits[range(args.batch_size), actions] - torch.logsumexp(sum_logits, dim=-1)
        p_backward_sum += torch.log(torch.tensor(1 / (i + 1))).to(args.device)

    log_rewards = args.reward_exponent * batch_log_rewards(batch[:, 1:], M, args.k).to(args.device).detach()
    loss = (log_Z.sum() + p_forward_sum - p_backward_sum - log_rewards).pow(2).mean() 
    loss.backward()
    optimizer.step()
    Z_optimizer.step()

    assert batch[:, 1:].max() < 2 ** args.k
    return loss.cpu().item(), batch[:, 1:].cpu()


def DB_train_step(model, optimizer, M, args):
    # This code is pretty simple because all trajectories in our graph have the same length.
    model.train()
    optimizer.zero_grad()

    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    batch = torch.tensor([[2 ** args.k + 1] + ([2 ** args.k] * (args.n // args.k)) for i in range(args.batch_size)]).to(args.device)
    loss = torch.tensor(0.0).to(args.device)
    pred_logits = None
    pred_f = None

    for i in range(args.n // args.k):
        pos_mask = batch != 2 ** args.k
        all_logits = model(batch.T)
        pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)
        log_f = all_logits[0, :, 2**args.k]

        if pred_logits is not None:
            loss += (pred_f + pred_logits - log_f - torch.log(torch.tensor(1 / i)).to(args.device)).pow(2).mean()

        with torch.no_grad():
            _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

            actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args)

            batch_cl = batch.clone()
            batch_cl[range(args.batch_size), positions] = words
            batch = batch_cl
 
        pred_logits = sum_logits[range(args.batch_size), actions] - torch.logsumexp(sum_logits, dim=-1)
        pred_f = log_f

    log_rewards = args.reward_exponent * batch_log_rewards(batch[:, 1:], M, args.k).to(args.device).detach()
    loss += (pred_f + pred_logits - log_rewards - torch.log(torch.tensor(1 / (args.n // args.k))).to(args.device)).pow(2).mean()
    loss = loss / (args.n // args.k)
    loss.backward()
    optimizer.step()
    
    assert batch[:, 1:].max() < 2 ** args.k
    return loss.cpu().item(), batch[:, 1:].cpu()


def SubTB_train_step(model, optimizer, M, args):
    # This code is pretty simple because all trajectories in our graph have the same length.
    model.train()
    optimizer.zero_grad()

    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    batch = torch.tensor([[2 ** args.k + 1] + ([2 ** args.k] * (args.n // args.k)) for i in range(args.batch_size)]).to(args.device)
    log_pfs = torch.zeros(args.n // args.k + 1, args.batch_size).to(args.device)
    log_pbs = torch.zeros(args.n // args.k + 1, args.batch_size).to(args.device)
    log_fs = torch.zeros(args.n // args.k + 1, args.batch_size).to(args.device)

    for i in range(args.n // args.k):
        pos_mask = batch != 2 ** args.k
        all_logits = model(batch.T)
        pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)
        log_fs[i] = all_logits[0, :, 2**args.k]

        with torch.no_grad():
            _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

            actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args)

            batch_cl = batch.clone()
            batch_cl[range(args.batch_size), positions] = words
            batch = batch_cl
 
        log_pfs[i] = sum_logits[range(args.batch_size), actions] - torch.logsumexp(sum_logits, dim=-1)

        if (i > 0):
            log_pbs[i] =  torch.log(torch.tensor(1 / i).to(args.device))

    log_fs[-1] = args.reward_exponent * batch_log_rewards(batch[:, 1:], M, args.k).to(args.device).detach()
    log_pbs[-1] = torch.log(torch.tensor(1 / (args.n // args.k))).to(args.device)

    loss = torch.tensor(0.0).to(args.device)
    total_lambda = torch.tensor(0.0).to(args.device)
    for i in range(log_fs.shape[0]):
        for j in range(i + 1, log_fs.shape[0]):
            lmbd = args.subtb_lambda ** (j - i)
            loss += lmbd * (log_fs[i, :] + log_pfs[i:j, :].sum(dim=0) - log_fs[j, :] - log_pbs[i+1:j+1, :].sum(dim=0)).pow(2).mean()
            total_lambda += lmbd
    loss /= total_lambda
    
    loss.backward()
    optimizer.step()
    
    assert batch[:, 1:].max() < 2 ** args.k
    return loss.cpu().item(), batch[:, 1:].cpu()
 

def SoftQL_collect_experience(rb, model, target_model, M, args):
    # This code is pretty simple because all trajectories in our graph have the same length.

    # The seqence has length n/k + 1 and at the beginning looks like [2^k + 1, 2^k, 2^k, ..., 2^k].
    # 2^k + 1: [BOS] token, 2^k: token for "empty" word.
    batch = torch.tensor([[2 ** args.k + 1] + ([2 ** args.k] * (args.n // args.k)) for i in range(args.batch_size)]).to(args.device)
    with torch.no_grad():
        for i in range(args.n // args.k):
            pos_mask = batch != 2 ** args.k
        
            all_logits = model(batch.T)
            _, _, sum_logits = process_logits(all_logits, pos_mask, args)
            _, _, sum_uniform = process_logits(0.0 * all_logits.clone(), pos_mask, args)

            actions, positions, words = sample_forward(sum_logits, sum_uniform, batch, args)

            next_batch = batch.clone()
            next_batch[range(args.batch_size), positions] = words
            rewards = torch.log(torch.tensor([1 / (i+1)] * args.batch_size).to(args.device)) 

            # The last added word
            if i + 1 == args.n // args.k:
                rewards += args.reward_exponent * batch_log_rewards(next_batch[:, 1:], M, args.k).to(args.device)
                is_done = torch.tensor([1.0] * args.batch_size).to(args.device)
            else:
                is_done = torch.tensor([0.0] * args.batch_size).to(args.device)

            rb_record = TensorDict(
                {
                    "state": batch,
                    "action": actions,
                    "next_state": next_batch,
                    "rewards": rewards,
                    "is_done": is_done,
                }, 
                batch_size=args.batch_size
            )
            rb.extend(rb_record) # add record to replay buffer
            batch = next_batch

    assert batch[:, 1:].max() < 2 ** args.k
    return batch[:, 1:].cpu()
        

def SoftQL_learn_rb(progress, rb, model, target_model, optimizer, M, args):
    # Select loss function
    if args.softql_loss == 'Huber':
        loss_fn = torch.nn.HuberLoss(reduction='none')
    else:
        loss_fn = torch.nn.MSELoss(reduction='none')
    if args.anneal_per_beta:
        # Update beta parameter of experience replay
        add_beta = (1. - args.per_beta) * progress
        rb._sampler._beta = args.per_beta + add_beta

    model.train()
    optimizer.zero_grad()

    # Sample from replay buffer
    rb_batch = rb.sample().to(args.device)
    # Compute td-loss
    pos_mask = rb_batch["state"] != 2 ** args.k
    all_logits = model(rb_batch["state"].T)
    _, _, sum_logits = process_logits(all_logits, pos_mask, args)
    if args.m_alpha > 0:
        all_target_logits = target_model(rb_batch["state"].T)
        _, _, sum_target_logits = process_logits(all_target_logits, pos_mask, args)
        norm_target_logits = sum_target_logits / args.entropy_coeff  

    q_values = sum_logits[range(args.rb_batch_size), rb_batch["action"]]
    
    with torch.no_grad():
        pos_mask = rb_batch["next_state"] != 2 ** args.k
        all_target_logits = target_model(rb_batch["next_state"].T)
        _, _, sum_target_logits = process_logits(all_target_logits, pos_mask, args)
        target_v_next_values = args.entropy_coeff * torch.logsumexp(sum_target_logits / args.entropy_coeff, dim=-1)
        target_v_next_values[rb_batch["is_done"].bool()] = 0.0
        td_target = rb_batch["rewards"] + target_v_next_values
        
        if args.m_alpha > 0:
            target_log_policy = norm_target_logits[range(args.rb_batch_size), rb_batch["action"]] - torch.logsumexp(norm_target_logits, dim=-1)
            munchausen_penalty = torch.clamp(
                args.entropy_coeff * target_log_policy,
                min=args.m_l0, max=1
            )
            td_target += args.m_alpha * munchausen_penalty
    
    td_errors = loss_fn(q_values, td_target)
    td_errors[rb_batch["is_done"].bool()] *= args.leaf_coeff

    # Update PER
    rb_batch["td_error"] = td_errors
    rb.update_tensordict_priority(rb_batch)

    # Compute loss with IS correction
    loss = (td_errors * rb_batch["_weight"]).mean()
    #loss = td_errors.mean()
    loss.backward()
    optimizer.step()

    return loss.cpu().item()


def compute_correlation(model, M, test_set, args, rounds=10, batch_size=180):
    # Sampling a trajectory from PB(tau | x) when PB is uniform over parents 
    # in this case is equvalent to starting at s0 and randomly choosing the order 
    # in which we replace empty words with words at corresponding positions from x.
    # Thus we can sample trajectories and compute PF(tau) in parallel.
    model.eval()
    assert len(test_set) % batch_size == 0
    p_forward_sums = torch.zeros(len(test_set), rounds).to(args.device)

    for round in range(rounds):
        for batch_idx in range(len(test_set) // batch_size):
            batch = torch.tensor([[2 ** args.k + 1] + ([2 ** args.k] * (args.n // args.k)) for i in range(batch_size)]).to(args.device)
            for i in range(args.n // args.k):
                with torch.no_grad():
                    pos_mask = batch != 2 ** args.k
                    all_logits = model(batch.T)
                    pos_logits, word_logits, sum_logits = process_logits(all_logits, pos_mask, args)

                    # There is a bug in pytorch that allows to sample objects that has 0 probability (happens very rarely but still happens).
                    # This loop basically resamples until everything is correct.
                    while True:
                        uniform_probs = torch.zeros(batch_size, args.n // args.k + 1) + 1 / (args.n // args.k - i)
                        uniform_probs[pos_mask] = 0.0
                        positions = Categorical(probs=uniform_probs).sample().to(args.device)
                        if (batch[range(batch_size), positions] == 2 ** args.k).sum() == batch_size:
                            break

                    assert positions.min() >= 1
                    assert positions.max() <= args.n // args.k 

                    words = []
                    for j in range(batch_size):
                        s = test_set[batch_idx * batch_size + j]
                        word = int(s[(positions[j] - 1) * args.k:positions[j] * args.k], base=2)
                        words.append(word)
                    words = torch.tensor(words).to(args.device)
                    
                    batch_cl = batch.clone()
                    batch_cl[range(batch_size), positions] = words
                    batch = batch_cl

                    actions = positions * (2 ** args.k) + words
                    log_pf = sum_logits[range(batch_size), actions] / args.entropy_coeff - torch.logsumexp(sum_logits / args.entropy_coeff, dim=-1)
                    p_forward_sums[batch_idx * batch_size:(batch_idx + 1) * batch_size, round] += log_pf

    p_forward_sum = torch.logsumexp(p_forward_sums, dim=-1)
    log_rewards = np.array([log_reward(s, M) for s in test_set])
    return spearmanr((args.reward_exponent * log_rewards), (p_forward_sum.detach().cpu().numpy()))


def main(args):
    torch.manual_seed(args.seed)
    device = args.device
    assert args.n % args.k == 0
    H = ["00000000", "11111111", "11110000", "00001111", "00111100"]
    assert args.n % len(H[0]) == 0
    M = construct_M(args.n, len(H[0]), H, args.M_size, seed=args.seed)
    test_set = construct_test_set(M, seed=args.seed)
    print(f"test set size: {len(test_set)}")

    model = TransformerModel(ntoken=2**args.k+2, d_model=64, d_hid=64, nhead=8, nlayers=3, 
                             seq_len=args.n//args.k, dropout=args.dropout).to(device)
    if args.objective == "softql":
        target_model = TransformerModel(ntoken=2**args.k+2, d_model=64, d_hid=64, nhead=8, nlayers=3, 
                                        seq_len=args.n//args.k, dropout=args.dropout).to(device)
        target_model.load_state_dict(model.state_dict())
        
    log_Z = nn.Parameter(torch.tensor(np.ones(64) * 0.0 / 64, requires_grad=True, device=device))
    
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=1e-5)
    Z_optimizer = torch.optim.Adam([log_Z], args.z_learning_rate, weight_decay=1e-5)

    rb =  TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.rb_size),
        sampler=PrioritizedSampler(
            max_capacity=args.rb_size,
            alpha=args.per_alpha,
            beta=args.per_beta,
        ),
        batch_size=args.rb_batch_size,
        priority_key="td_error"
    )
    
    modes = [False] * len(M)
    avg_reward = 0.0

    corr_nums = []
    mode_nums = []
    if args.objective == "softql":
        # Renormalize entropy for Munchausen DQN
        args.entropy_coeff *= 1/(1 - args.m_alpha)
    
    for it in range(args.num_iterations + 1):
        progress = float(it) / args.num_iterations
        if args.objective == "tb":
            loss, batch = TB_train_step(model, log_Z, optimizer, Z_optimizer, M, args)
        elif args.objective == "db":
            loss, batch = DB_train_step(model, optimizer, M, args)
        elif args.objective == "subtb":
            loss, batch = SubTB_train_step(model, optimizer, M, args)
        elif args.objective == "softql":
            # First, collect experiences for experience replay
            batch = SoftQL_collect_experience(rb, model, target_model, M, args)
            # Next, sample transitions from the buffer and calculate the loss
            if it > args.start_learning:
                loss = SoftQL_learn_rb(progress, rb, model, target_model, optimizer, M, args)
            else:
                loss = 0.0

            if it % args.update_target_every == 0:
                target_model.load_state_dict(model.state_dict())
        
        avg_reward += (batch_rewards(batch, M, args.k) ** args.reward_exponent).sum().item() / args.batch_size

        batch_strings = [token_seq_to_str(seq, args.k) for seq in batch]
        for m in range(len(M)):
            if modes[m]:
                continue
            for i in range(args.batch_size):
                if distance(M[m], batch_strings[i]) <= args.mode_threshold:
                    modes[m] = True
                    break
        
        if it > 0 and it % args.print_every == 0:
            print(f"{it}, loss: {loss}, modes: {sum(modes)}, avg reward: {avg_reward / args.print_every}, log_Z: {log_Z.sum().cpu().item()}")
            avg_reward = 0.0

        if it > 0 and it % 2000 == 0:
            if args.print_modes:
                print("found modes:")
                for m in range(len(M)):
                    if modes[m]:
                        print(M[m])
            mode_nums.append(sum(modes))

            sp_corr = compute_correlation(model, M, test_set, args, rounds=args.corr_num_rounds)
            print(f"test set reward correlation: {sp_corr}")
            corr_nums.append(sp_corr.statistic)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
