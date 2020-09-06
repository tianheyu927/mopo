from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'halfcheetah',
    'task': 'medium-v0',
    'exp_name': 'halfcheetah_medium'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/halfcheetah-medium-v0',
    'pool_load_max_size': 10**6,
    'rollout_length': 1,
    'penalty_coeff': 1.0
})