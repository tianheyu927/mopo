from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'walker2d',
    'task': 'medium-replay-v0',
    'exp_name': 'walker2d_medium_replay'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/walker2d-medium-replay-v0',
    'pool_load_max_size': 100930,
    'rollout_length': 1,
    'penalty_coeff': 1.0
})