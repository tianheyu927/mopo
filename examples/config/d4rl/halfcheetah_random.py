from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'halfcheetah',
    'task': 'random-v0',
    'exp_name': 'halfcheetah_random'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/halfcheetah-random-v0',
    'pool_load_max_size': 10**6,
    'rollout_length': 5,
    'penalty_coeff': 0.5
})