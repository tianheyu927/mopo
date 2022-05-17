from .base_mopo import mopo_params, deepcopy

params = deepcopy(mopo_params)
params.update({
    'domain': 'HalfCheetah',
    'task': 'v2',
    'exp_name': 'halfcheetah_dogo'
})
params['kwargs'].update({
    'pool_load_path': '/home/ajc348/rds/hpc-work/mopo/rollouts/softlearning/HalfCheetah/v3/2022-05-16T12-29-56-my-sac-experiment-1/id=8592b_00000-seed=4378/combined_transitions.npy',
    'pool_load_max_size': 101000,
    'rollout_length': 5,
    'penalty_coeff': 1.0
})
