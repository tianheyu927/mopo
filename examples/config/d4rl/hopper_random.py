from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'hopper',
    'task': 'random-v0',
    'exp_name': 'hopper_random'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/hopper-random-v0',
    'pool_load_max_size': 10**6,
    'rollout_length': 5,
    'penalty_coeff': 1.0
})