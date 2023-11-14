from ml_collections.config_dict import ConfigDict


def get_config(seed : str):
    config = ConfigDict(
        {
            'seed': int(seed),
            'device': 'cpu',
            'validation_interval': 100,
            'validation_samples': 200000,
            'wandb_project': '',    # if empty, do not use wandb
            'n_envs': 16,
            'n_trajectories': 1000000
        }
    )

    return config
