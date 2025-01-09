"""

CONFIG = {
    "lr": 1e-3,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay": 500,
    "episodes": 500,
    "buffer_size": 10000,
    "batch_size": 64,
    "target_update": 10,
    "max_steps": 1000,
}

"""
CONFIG = {
    "lr": 1e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_min": 0.1,
    "epsilon_decay": 0.995,
    "episodes": 500,
    "buffer_size": 100000,
    "batch_size": 32,
    "target_update_freq": 1000,
    "max_steps": 1000,
    "epsilon_end": 0.1,
    "steps_per_episode": 10000
}
