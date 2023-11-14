import numpy as np
import torch
try:
    import wandb
except ModuleNotFoundError:
    pass
from tqdm import tqdm, trange


from algorithms import SACGFlowNet, TorchRLReplayBuffer

from gfn.modules import DiscretePolicyEstimator
from experiments.utils import validate
from gfn.utils.modules import NeuralNet, DiscreteUniform
from gfn.env import Env


from ml_collections.config_dict import ConfigDict


def train_sac(
        env: Env,
        experiment_name: str,
        general_config: ConfigDict,
        algo_config: ConfigDict):

    if algo_config.uniform_pb:
        experiment_name += '_uniform-pb'
    else:
        experiment_name += '_learnt-pb'

    #if algo_config.is_double:
        #experiment_name += '_Double'

    if algo_config.replay_buffer.replay_buffer_size > 0:
        if algo_config.replay_buffer.prioritized:
            experiment_name += '_PER'
        else:
            experiment_name += '_ER'

    if algo_config.loss_type != 'MSE':
        experiment_name += f'_loss_type={algo_config.loss_type}'

    #if algo_config.munchausen.alpha > 0:
        #experiment_name += f"_M_alpha={algo_config.munchausen.alpha}"

    use_wandb = len(general_config.wandb_project) > 0
    policy_module = NeuralNet(
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
            torso=policy_module.torso if algo_config.tied else None,
        )

    policy_estimator = DiscretePolicyEstimator(
        env=env, module=policy_module, forward=True)
    pb_estimator = DiscretePolicyEstimator(
        env=env, module=pb_module, forward=False)

    q1_module = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
        torso=policy_module.torso   # TODO: make it a parameter
    )
    q2_module = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
        torso=policy_module.torso   # TODO: make it a parameter
    )

    q1_target_module = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
    )
    q2_target_module = NeuralNet(
        input_dim=env.preprocessor.output_dim,
        output_dim=env.n_actions,
        hidden_dim=algo_config.net.hidden_dim,
        n_hidden_layers=algo_config.net.n_hidden,
    )

    replay_buffer_size = algo_config.replay_buffer.replay_buffer_size

    entropy_coeff = 1  #/(1 - algo_config.munchausen.alpha)  # to make (1-alpha)*tau=1
    gflownet = SACGFlowNet(
        actor=policy_estimator,
        q1=DiscretePolicyEstimator(
            env=env, module=q1_module, forward=True
        ),
        q2=DiscretePolicyEstimator(
            env=env, module=q2_module, forward=True
        ),
        q1_target=DiscretePolicyEstimator(
            env=env, module=q1_target_module, forward=True
        ),
        q2_target=DiscretePolicyEstimator(
            env=env, module=q2_target_module, forward=True
        ),
        pb=pb_estimator,
        on_policy=True if replay_buffer_size == 0 else False,
        entropy_coeff=entropy_coeff
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

    #print(dict(gflownet.named_parameters()))
    params = [
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items()
                if ('actor.' in k) or ('pb.' in k)
            ],
            "lr": algo_config.policy_learning_rate,
            "eps": algo_config.adam_eps
        },
        {
            "params": [
                v for k, v in dict(gflownet.named_parameters()).items()
                if ('q1.' in k) or ('q2.' in k)
            ],
            "lr": algo_config.q_learning_rate,
            "eps": algo_config.adam_eps
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
                    preds1, preds2, targets = gflownet.get_td_preds_target(
                        training_samples
                    )
                    td_error = loss_fn(preds1, targets) + loss_fn(preds2, targets)
                    replay_buffer.add(training_samples, td_error)
                    # Annealing of beta
                    replay_buffer.update_beta(progress)
                else:
                    replay_buffer.add(training_samples)

            if iteration > algo_config.learning_starts:
                training_objects, rb_batch = replay_buffer.sample()
                preds1, preds2, targets = gflownet.get_td_preds_target(training_objects)
        else:
            training_objects = training_samples
            preds1, preds2, targets = gflownet.get_td_preds_target(training_objects)

        if iteration > algo_config.learning_starts:
            optimizer.zero_grad()
            td_error = loss_fn(preds1, targets) + loss_fn(preds2, targets)
            if replay_buffer is not None and replay_buffer.prioritized:
                replay_buffer.update_priority(rb_batch, td_error.detach())
            # Compute both policy loss and q-value loss
            loss = td_error.mean() + gflownet.policy_loss(training_objects)
            loss.backward()
            optimizer.step()

        visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)

        to_log = {"states_visited": states_visited}
        if iteration > algo_config.learning_starts:
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
