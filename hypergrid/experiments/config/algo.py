from ml_collections.config_dict import ConfigDict


def get_config(alg_name):
    algo_config = {

        "soft_ql": ConfigDict({
                'name': 'SoftQL',
                'tied': False,
                'uniform_pb': True,
                'learning_rate': 1e-3,
                'loss_type': "Huber",  # "MSE" or "Huber"
                'net': ConfigDict({
                    'hidden_dim': 256,
                    'n_hidden': 2
                }),
                'replay_buffer': ConfigDict({
                    'replay_buffer_size': 100000,
                    'batch_size': 256,
                    'prioritized': True,
                    'alpha': 0.5,
                    'beta': 0.0
                }),
                'munchausen': ConfigDict({
                    'alpha': 0.00,
                    'l0': 0
                }),
                'learning_starts': 16,
                'update_frequency': 1,
                'target_network_frequency': 1,
                'tau': 0.25,
                'is_double': False
            }),

        "munchausen_ql": ConfigDict({
                'name': 'SoftQL',
                'tied': False,
                'uniform_pb': True,
                'learning_rate': 1e-3,
                'loss_type': "Huber",  # "MSE" or "Huber"
                'net': ConfigDict({
                    'hidden_dim': 256,
                    'n_hidden': 2
                }),
                'replay_buffer': ConfigDict({
                    'replay_buffer_size': 100000,
                    'batch_size': 256,
                    'prioritized': True,
                    'alpha': 0.5,
                    'beta': 0.0
                }),
                'munchausen': ConfigDict({
                    'alpha': 0.15,
                    'l0': -100
                }),
                'learning_starts': 16,
                'update_frequency': 1,
                'target_network_frequency': 1,
                'tau': 0.25,
                'is_double': False
            }),

        "sac": ConfigDict({
                'name': 'SAC',
                'tied': True,
                'uniform_pb': False,
                'q_learning_rate': 1e-2,
                'policy_learning_rate': 1e-3,
                'adam_eps' : 1e-4,
                'loss_type': "Huber",  # "MSE" or "Huber"
                'net': ConfigDict({
                    'hidden_dim': 256,
                    'n_hidden': 2
                }),
                'replay_buffer': ConfigDict({
                    'replay_buffer_size': 100000,
                    'batch_size': 256,
                    'prioritized': True,
                    'alpha': 0.5,
                    'beta': 0.0
                }),
                'learning_starts': 16,
                'target_network_frequency': 1,
                'tau': 0.25,
            }),

        "tb": ConfigDict({
            'name': 'TrajectoryBalance',
            'tied': True,
            'uniform_pb': False,
            'learning_rate': 1e-3,
            'learning_rate_Z': 1e-1,
            'net': ConfigDict({
                'hidden_dim': 256,
                'n_hidden': 2
            }),
            'replay_buffer_size': 0
        }),

        "db": ConfigDict({
            'name': 'DetailedBalance',
            'tied': True,
            'uniform_pb': False,
            'learning_rate': 1e-3,
            'net': ConfigDict({
                'hidden_dim': 256,
                'n_hidden': 2
            }),
            'replay_buffer_size': 0
        }),

        "subtb": ConfigDict({
            'name': 'SubTrajectoryBalance',
            'tied': True,
            'uniform_pb': False,
            'learning_rate': 1e-3,
            'net': ConfigDict({
                'hidden_dim': 256,
                'n_hidden': 2
            }),
            'replay_buffer_size': 0,
            'subTB_weighting' : "geometric_within",
            'subTB_lambda': 0.9,
        }),

        "perfect": ConfigDict({
            'name': 'GroundTruth'
        }),

        "uniform": ConfigDict({
            'name': 'Uniform'
        })
    }

    return algo_config[alg_name]
