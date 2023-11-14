import numpy as np
try:
    import wandb
except ModuleNotFoundError:
    pass
from tqdm import tqdm, trange


from gfn.modules import DiscretePolicyEstimator
from experiments.utils import validate
from gfn.utils.modules import DiscreteUniform
from gfn.env import Env
from gfn.samplers import Sampler

from ml_collections.config_dict import ConfigDict


def train_uniform(
        env: Env,
        experiment_name: str,
        general_config: ConfigDict,
        algo_config: ConfigDict):

    use_wandb = len(general_config.wandb_project) > 0
    pf_module = DiscreteUniform(env.n_actions)
    pf_estimator = DiscretePolicyEstimator(
        env=env, module=pf_module, forward=True)
    sampler = Sampler(estimator=pf_estimator)

    visited_terminating_states = env.States.from_batch_shape((0,))

    states_visited = 0
    kl_history, l1_history, nstates_history = [], [], []

    # Train loop
    n_iterations = general_config.n_trajectories // general_config.n_envs
    for iteration in trange(n_iterations):
        trajectories = sampler.sample_trajectories(n_trajectories=general_config.n_envs)
        visited_terminating_states.extend(trajectories.last_states)

        states_visited += len(trajectories)

        to_log = {"states_visited": states_visited}
        to_log.update({"traj_len" : trajectories.when_is_done.float().mean().item()})

        if use_wandb:
            wandb.log(to_log, step=iteration)

        if (iteration + 1) % general_config.validation_interval == 0:
            validation_info = validate(
                env,
                None,
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
