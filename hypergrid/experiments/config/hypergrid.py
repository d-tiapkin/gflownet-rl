from ml_collections.config_dict import ConfigDict


def get_config(env_type):

    env_config = {
        "standard" : ConfigDict({
            'name': 'Hypergrid',
            'reward_type': 'standard',
            'ndim': 4,
            'height': 20,
            'R0': 0.001,
            'R1': 0.5,
            'R2': 2.0
        }),
        "hard" : ConfigDict({
            'name': 'Hypergrid',
            'reward_type': 'hard',
            'ndim': 4,
            'height': 20,
            'R0': 0.0001,
            'R1': 1.0,
            'R2': 3.0
        })
    }

    return env_config[env_type]
