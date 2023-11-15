import hydra
import gzip
import os
import pickle
import time
import warnings
from copy import deepcopy

from omegaconf import OmegaConf
import logging

import numpy as np
import torch

from mol_mdp_ext import BlockMoleculeDataExtended

from torchrl.data import ReplayBuffer, ListStorage
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

from utils.dataset import Dataset
from utils.proxy import make_model, Proxy
from utils.metrics import compute_correlation
from utils.chem import compute_num_of_modes


warnings.filterwarnings('ignore')

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


class SoftRLDataset(Dataset):
    def __init__(self, env_args, algo_args, bpath, device, floatX=torch.double):
        super().__init__(env_args, bpath, device, floatX)
        self.initial_beta = algo_args.per_beta
        self.rb = ReplayBuffer(
            storage=ListStorage(max_size=algo_args.rb_size),
            sampler=PrioritizedSampler(
                max_capacity=algo_args.rb_size,
                alpha=algo_args.per_alpha,
                beta=algo_args.per_beta
            ),
            collate_fn=lambda x: x,      # To store molecule data
            batch_size=algo_args.rb_batch
        )
        # Entropy coeff for sampling since pi_\theta is not just softmax for M-DQN
        self.entropy_coeff = algo_args.entropy_coeff
        # Dueling architecture
        self.dueling = algo_args.dueling
    
    def inverse_r2r(self, reward):
        # This inverse transform works only for reward > R_min
        return self.reward_norm * reward**(1/self.reward_exp)

    def set_target_model(self, model):
        self.target_model = model

    def _get_sample_model(self):
        m = BlockMoleculeDataExtended()
        samples = [] # (state, action, soft-rl reward, next state, done)
        
        # For statistics
        mdp_rewards = []

        max_blocks = self.max_blocks
        if self.early_stop_reg > 0 and np.random.uniform() < self.early_stop_reg:
            early_stop_at = np.random.randint(self.min_blocks, self.max_blocks + 1)
        else:
            early_stop_at = max_blocks + 1

        trajectory_stats = []
        # max_blocks is a maximum length of trajecotry
        for t in range(max_blocks):
            s = self.mdp.mols2batch([self.mdp.mol2repr(m)])
            with torch.no_grad():
                s_o, m_o = self.sampling_model(s)
            ## fix from run 330 onwards
            if t < self.min_blocks:
                m_o = m_o * 0 - 1000 # prevent assigning prob to stop
                                     # when we can't stop

            # if archtechture is dueling, it is advantage, not q-value, however it is not change a distribution of policy
            logits = torch.cat([m_o[:, 0].reshape(-1), s_o.reshape(-1)])  
            policy = torch.distributions.Categorical(
                logits=logits / self.entropy_coeff
            )
            action = policy.sample().item()
            # Epsilon greedy policy
            if self.random_action_prob > 0 and self.train_rng.uniform() < self.random_action_prob:
                action = self.train_rng.randint(int(t < self.min_blocks), logits.shape[0])
            if t == early_stop_at:
                action = 0

            # For PER and statistics
            if self.dueling:
                # Q_value = value + advantage - LSE_lambda(advantage) = value + log_policy 
                # (that is just logits because of inner normalization in torch)
                q_value = policy.logits + m_o[:,1]
                # value is just value
                v_value = m_o[:,1]
            else:
                # q_value is already computed
                q_value = logits[action].item()
                # valus = LSE_lambda(v_value)
                v_value = self.entropy_coeff * torch.logsumexp(logits / self.entropy_coeff, 0).item()

            state = m
            next_state = BlockMoleculeDataExtended() # default: empty molecule
            mdp_reward = None
            done = 0

            if t >= self.min_blocks and action == 0:
                r = self._get_reward(m)
                # A "terminal copy" of this mol has a unique parent
                mdp_reward = np.log(r)  
                action = (-1,0)
                done = 1
            else:
                action = max(0, action-1)
                action = (action % self.mdp.num_blocks, action // self.mdp.num_blocks)
                m = self.mdp.add_block_to(m, *action)
                next_state = m
                if len(m.blocks) and not len(m.stems) or t == max_blocks - 1:
                    # can't add anything more to this mol so let's make it
                    # terminal. Note that this node's parent isn't just m,
                    # because this is a sink for all parent transitions
                    r = self._get_reward(m)
                    mdp_reward = np.log(r) + np.log(1/n)
                    done = 1
                else:
                    n = len(self.mdp.parents(m))
                    # MDP-reward is equal to log(1/n)
                    mdp_reward = np.log(1/n)
            
            next_exit_banned = t+1 < self.min_blocks
            samples.append((state, action, mdp_reward, next_state, done, next_exit_banned))

            mdp_rewards.append(mdp_reward)
            trajectory_stats.append((q_value, action, v_value))

            if done:
                break

        self.sampled_mols.append((self.inverse_r2r(r), m, trajectory_stats, None))

        # Assumes only PER reward model
        m.reward = r
        self.rb.extend(samples)

        return samples

    def sample(self, n):
        trajectories = [self._get_sample_model() for i in range(n)]
        batch = [*zip(*sum(trajectories, []))]
        return batch
    
    def sample2batch(self, mb, process=False):
        if not process:
            # Return states to compute mode-related metrics
            return []
        states, actions, mdp_rewards, next_states, dones, next_exit_banned = mb
        mols = (states,)
        states = self.mdp.mols2batch([self.mdp.mol2repr(s) for s in states])
        actions = torch.tensor(actions, device=self._device).long()
        mdp_rewards = torch.tensor(mdp_rewards, device=self._device).to(self.floatX)
        next_states = self.mdp.mols2batch([self.mdp.mol2repr(s) for s in next_states])
        dones = torch.tensor(dones, device=self._device).to(self.floatX)
        next_exit_banned = torch.tensor(next_exit_banned, device=self._device).bool()
        return states, actions, mdp_rewards, next_states, dones, next_exit_banned, mols

    def rb_sample(self):
        mb, info = self.rb.sample(return_info=True)
        index = info['index']
        # PER Importance sampling weights
        weight = torch.tensor(info['_weight'], device=self._device).to(self.floatX)

        batch = self.sample2batch([*zip(*mb)], process=True)
        return batch, index, weight
    
    def update_priority(self, index, priority):
        self.rb.update_priority(index, priority)

    def update_beta(self, progress: float) -> None:
        add_beta = (1. - self.initial_beta) * progress
        self.rb._sampler._beta = self.initial_beta + add_beta


def train_model_with_proxy(args, model, proxy, dataset, bpath, test_mols_path, target_model=None, num_steps=None, do_save=True):
    log.info('Start training!')
    device = torch.device('cuda')

    if num_steps is None:
        num_steps = args.num_iterations + 1
    
    # Rescale entropy coeff to compensate Munchausen penalty
    args.entropy_coeff *= 1/(1 - args.m_alpha)

    tau = args.bootstrap_tau
    start_learning = max(args.start_learning, args.rb_batch)

    if args.loss_fn == 'MSE':
        loss_fn = torch.nn.MSELoss(reduction='none')
    elif args.loss_fn == 'Huber':
        loss_fn = torch.nn.HuberLoss(reduction='none')
    else:
        raise ValueError('Unknown Soft-DQN loss type')

    if do_save:
        exp_dir = './hydra_mol_data'
        os.makedirs(exp_dir, exist_ok=True)

    dataset.set_sampling_model(model, proxy, sample_prob=args.sample_prob)
    dataset.set_target_model(target_model)

    def save_stuff(iter):
        corr_logp = compute_correlation(model, dataset.mdp, bpath, test_mols_path, entropy_coeff=args.entropy_coeff)
        pickle.dump(corr_logp, gzip.open(f'{exp_dir}/{iter}_model_logp_pred.pkl.gz', 'wb'))

        pickle.dump([i.data.cpu().numpy() for i in model.parameters()],
                    gzip.open(f'{exp_dir}/' + str(iter) + '_params.pkl.gz', 'wb'))

        pickle.dump(dataset.sampled_mols,
                    gzip.open(f'{exp_dir}/' + str(iter) + '_sampled_mols.pkl.gz', 'wb'))

        pickle.dump({'train_losses': train_losses,
                     'test_losses': test_losses,
                     'test_infos': test_infos,
                     'time_start': time_start,
                     'time_now': time.time(),
                     'args': args,},
                    gzip.open(f'{exp_dir}/' + str(iter) + '_info.pkl.gz', 'wb'))

        pickle.dump(train_infos,
                    gzip.open(f'{exp_dir}/' + str(iter) + '_train_info.pkl.gz', 'wb'))
        true_log_r = [np.log(corr_logp[i][0][0]) for i in range(len(corr_logp))]
        pred_log_r = [corr_logp[i][1] for i in range(len(corr_logp))]
        return true_log_r, pred_log_r
        
    opt = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay,
                           betas=(args.opt_beta, args.opt_beta2),
                           eps=args.opt_epsilon)

    mbsize = args.mbsize

    # To initiate random sampling
    dataset.random_action_prob = 1.0

    last_losses = []
    train_losses = []
    test_losses = []
    test_infos = []
    train_infos = []
    time_start = time.time()
    time_last_check = time.time()


    for i in range(num_steps):
        progress = float(i) / num_steps
        _ = dataset.sample2batch(dataset.sample(mbsize))
        
        # Train loop
        if len(dataset.rb) < start_learning:
            # Wait with uniform sampling
            continue
        else:
            # Linear annealing of epsilon
            linear_scheduling_a = (1 - args.random_action_prob) / args.exploration_fraction
            linear_scheduling_eps = 1 - linear_scheduling_a * progress
            dataset.random_action_prob = max(args.random_action_prob, linear_scheduling_eps)
        
        if i % args.update_every_traj == 0:
            minibatch, index, weight = dataset.rb_sample()
            states, actions, mdp_rewards, next_states, dones, next_exit_banned, mols = minibatch
            copy_states = deepcopy(states)

            stem_out_s, mol_out_s = model(states, None)
            if args.dueling:
                q_states = -model.action_negloglikelihood(states, actions, 0, stem_out_s, mol_out_s) + mol_out_s[:, 1]
            else:
                q_states = model.index_output_by_action(states, stem_out_s, mol_out_s.squeeze(-1), actions)

            with torch.no_grad():
                stem_out_s_target, mol_out_s_target = target_model(next_states, None)
                if args.dueling:
                    # In the case of dueling architecture we already have output for value
                    target_v_nstates = mol_out_s_target[:, -1]
                else:
                    # Masking out impossible action
                    mol_out_s_target[next_exit_banned.bool(), 0] = 0 * mol_out_s_target[next_exit_banned.bool()] - 1000
                    # Computing value
                    target_v_nstates = args.entropy_coeff * target_model.out_to_lse(
                        next_states,
                        stem_out_s_target / args.entropy_coeff,
                        mol_out_s_target / args.entropy_coeff
                    )
                targets = mdp_rewards + (1. - dones) * target_v_nstates
                if args.m_alpha > 0:
                    target_stem_out_s, target_mol_out_s = target_model(copy_states, None)
                    target_log_policy = -target_model.action_negloglikelihood(
                        copy_states, actions, 0, target_stem_out_s, target_mol_out_s
                    )
                    munchausen_penalty = torch.clamp(
                        args.entropy_coeff * target_log_policy,
                        min=args.m_l0, max=1
                    )
                    targets += args.m_alpha * munchausen_penalty

            td_error = loss_fn(q_states, targets)
            # Reweights final transitions as in FM (if needed)
            if args.balanced_loss:
                td_error[dones.bool()] *= args.leaf_coef
            # Applying IS correction to final TD-loss
            loss = (td_error * weight).mean()
            opt.zero_grad()
            loss.backward()

            dataset.update_priority(index, td_error.detach().cpu())
            if args.anneal_per_beta:
                dataset.update_beta(progress)

            last_losses.append((loss.item(),))
            train_losses.append((loss.item(),))

            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)

            opt.step()
            model.training_steps = i + 1
            if args.update_target_every > 0 and i % args.update_target_every == 0:
                target_model.load_state_dict(model.state_dict())
            elif tau > 0:
                with torch.no_grad():
                    for _a,b in zip(model.parameters(), target_model.parameters()):
                        b.data.mul_(1-tau).add_(tau*_a)


        if not i % 100:
            last_losses = [np.round(np.mean(i), 3) for i in zip(*last_losses)]
            
            log.info(f"iter: {i}, loss: {last_losses}")
            log.info(f"grad norm:{np.array([p.grad.data.norm(2).item() for p in list(filter(lambda p: p.grad is not None, model.parameters()))]).mean()}")
            log.info(f"time: {time.time() - time_last_check}")

            time_last_check = time.time()
            last_losses = []

            if not i % args.save_every and do_save:
                # Save and compute metrics
                true_log_r, pred_log_r = save_stuff(i)
                log.info(f"correlation: {np.corrcoef(true_log_r, pred_log_r)[0][1]}")
                num_modes, num_above = compute_num_of_modes(dataset.sampled_mols, reward_thresh=7.0)
                log.info(f'num modes R > 7.0: {num_modes}, num candidates R > 7.0: {num_above}')

    true_log_r, pred_log_r = save_stuff(i)
    corr = np.corrcoef(true_log_r, pred_log_r)[0][1]
    log.info(f"final_correlation: {corr}")
    num_modes, num_above = compute_num_of_modes(dataset.sampled_mols, reward_thresh=7.0)
    log.info(f'num modes R > 7.0: {num_modes}, num candidates R > 7.0: {num_above}')
    log.info('End training!')
    return model

@hydra.main(config_path="configs", config_name="soft_dqn")
def run_soft_dqn(cfg):
    log.info(OmegaConf.to_yaml(cfg))
    bpath = cfg.environment.bpath
    device = torch.device('cuda')

    if cfg.environment.floatX == 'float32':
        floatX_fn = torch.float
    else:
        floatX_fn = torch.double
        
    dataset = SoftRLDataset(cfg.environment, cfg.algorithm, bpath, device, floatX=floatX_fn)
    mdp = dataset.mdp

    log.info("Making model...")
    model = make_model(cfg.environment, mdp, out_per_mol=1 + (1 if cfg.algorithm.dueling else 0))
    model.to(floatX_fn)
    model.to(device)

    if cfg.algorithm.bootstrap_tau > 0 or cfg.algorithm.update_target_every > 0:
        log.info("Making target model...")
        target_model = make_model(cfg.environment, mdp, out_per_mol=1 + (1 if cfg.algorithm.dueling else 0))
        target_model.to(floatX_fn)
        target_model.to(device)
        target_model.load_state_dict(model.state_dict())
    else:
        target_model = None

    log.info("Making proxy model...")
    proxy = Proxy(
        cfg.environment, 
        bpath, 
        device
    )

    return train_model_with_proxy(
        cfg.algorithm,
        model,
        proxy,
        dataset,
        cfg.environment.bpath,
        cfg.environment.test_mols_path,
        target_model=target_model,
        do_save=True
    )

if __name__ == '__main__':
    run_soft_dqn()