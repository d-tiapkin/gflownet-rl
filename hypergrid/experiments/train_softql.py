import numpy as np
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass
from tqdm import tqdm, trange


from algorithms import SoftDQNGFlowNet, TorchRLReplayBuffer

from gfn.modules import DiscretePolicyEstimator
from experiments.utils import validate
from gfn.utils.modules import NeuralNet, DiscreteUniform
from gfn.env import Env


from ml_collections.config_dict import ConfigDict


def train_softql(
        env: Env,
        experiment_name: str,
        general_config: ConfigDict,
        algo_config: ConfigDict):

    if algo_config.uniform_pb:
        experiment_name += '_uniform-pb'
    else:
        experiment_name += '_learnt-pb'

    if algo_config.update_frequency > 1:
        experiment_name += f"_freq={algo_config.update_frequency}"

    if algo_config.is_double:
        experiment_name += '_Double'

    if algo_config.replay_buffer.replay_buffer_size > 0:
        if algo_config.replay_buffer.prioritized:
            experiment_name += '_PER'
        else:
            experiment_name += '_ER'

    if algo_config.loss_type != 'MSE':
        experiment_name += f'_loss_type={algo_config.loss_type}'

    if algo_config.munchausen.alpha > 0:
        experiment_name += f"_M_alpha={algo_config.munchausen.alpha}"

    use_wandb = len(general_config.wandb_project) > 0
    pf_module = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
    )

    if algo_config.uniform_pb:
        pb_module = DiscreteUniform(env.n_actions - 1)
    else:
        pb_module = NeuralNet(
            input_dim=env.preprocessor.output_dim,
            output_dim=env.n_actions - 1,
            hidden_dim=algo_config.net.hidden_dim,
            n_hidden_layers=algo_config.net.n_hidden,
            torso=pf_module.torso if algo_config.tied else None,
        )

    pf_estimator = DiscretePolicyEstimator(
        env=env, module=pf_module, forward=True)
    pb_estimator = DiscretePolicyEstimator(
        env=env, module=pb_module, forward=False)

    pf_target = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
    )
    pf_target_estimator = DiscretePolicyEstimator(
        env=env, module=pf_target, forward=True)

    replay_buffer_size = algo_config.replay_buffer.replay_buffer_size

    entropy_coeff = 1/(1 - algo_config.munchausen.alpha)  # to make (1-alpha)*tau=1
    gflownet = SoftDQNGFlowNet(
        q=pf_estimator,
        q_target=pf_target_estimator,
        pb=pb_estimator,
        on_policy=True if replay_buffer_size == 0 else False,
        is_double=algo_config.is_double,
        entropy_coeff=entropy_coeff,
        munchausen_alpha=algo_config.munchausen.alpha,
        munchausen_l0=algo_config.munchausen.l0
    )

    replay_buffer = None
    if replay_buffer_size > 0:
        replay_buffer = TorchRLReplayBuffer(
            env,
            replay_buffer_size=replay_buffer_size,
            prioritized=algo_config.replay_buffer.prioritized,
            alpha=algo_config.replay_buffer.alpha,
            beta=algo_config.replay_buffer.beta,
            batch_size=algo_config.replay_buffer.batch_size
        )

    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items()
                if ("q_target" not in k)
            ],
            "lr": algo_config.learning_rate,
        }
    ]

    if algo_config.loss_type == 'MSE':
        loss_fn = torch.nn.MSELoss(reduction='none')
    elif algo_config.loss_type == 'Huber':  # Used for gradient clipping
        loss_fn = torch.nn.HuberLoss(reduction='none', delta=1.0)
    else:
        raise NotImplementedError(
            f"{algo_config.loss_type} loss is not supported"
        )

    optimizer = torch.optim.Adam(params)

    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    kl_history, l1_history, nstates_history = [], [], []

    # Train loop
    n_iterations = general_config.n_trajectories // general_config.n_envs
    for iteration in trange(n_iterations):
        progress = float(iteration) / n_iterations
        trajectories = gflownet.sample_trajectories(n_samples=general_config.n_envs)
        training_samples = gflownet.to_training_samples(trajectories)

        if replay_buffer is not None:
            with torch.no_grad():
                # For priortized RB
                if replay_buffer.prioritized:
                    scores = gflownet.get_scores(training_samples)
                    td_error = loss_fn(scores, torch.zeros_like(scores))
                    replay_buffer.add(training_samples, td_error)
                    # Annealing of beta
                    replay_buffer.update_beta(progress)
                else:
                    replay_buffer.add(training_samples)

            if iteration > algo_config.learning_starts:
                training_objects, rb_batch = replay_buffer.sample()
                scores = gflownet.get_scores(training_objects)
        else:
            training_objects = training_samples
            scores = gflownet.get_scores(training_objects)

        if iteration > algo_config.learning_starts and iteration % algo_config.update_frequency == 0:
            optimizer.zero_grad()
            td_error = loss_fn(scores, torch.zeros_like(scores))
            if replay_buffer is not None and replay_buffer.prioritized:
                replay_buffer.update_priority(rb_batch, td_error.detach())
            loss = td_error.mean()
            loss.backward()
            optimizer.step()

        visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)

        to_log = {"states_visited": states_visited}
        if iteration > algo_config.learning_starts and iteration % algo_config.update_frequency == 0:
            if iteration % algo_config.target_network_frequency == 0:
                gflownet.update_q_target(algo_config.tau)
            to_log.update({"loss" : loss.item()})

        if use_wandb:
            wandb.log(to_log, step=iteration)

        if (iteration + 1) % general_config.validation_interval == 0:
            validation_info = validate(
                env,
                gflownet,
                general_config.validation_samples,
                visited_terminating_states,
            )

            if use_wandb:
                wandb.log(validation_info, step=iteration)
            to_log.update(validation_info)
            tqdm.write(f"{iteration}: {to_log}")

            kl_history.append(to_log["kl_dist"])
            l1_history.append(to_log["l1_dist"])
            nstates_history.append(to_log["states_visited"])

        if (iteration + 1) % 1000 == 0:
            np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
            np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
            np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))

    np.save(f"{experiment_name}_kl.npy", np.array(kl_history))
    np.save(f"{experiment_name}_l1.npy", np.array(l1_history))
    np.save(f"{experiment_name}_nstates.npy", np.array(nstates_history))
