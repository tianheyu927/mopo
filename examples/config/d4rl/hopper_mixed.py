from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'hopper',
    'task': 'medium-replay-v0',
    'exp_name': 'hopper_medium_replay'
})
params['kwargs'].update({
    'pool_load_path': 'd4rl/hopper-mixed-v0',
    'pool_load_max_size': 200920,
    'rollout_length': 5,
    'penalty_coeff': 1.0,
})