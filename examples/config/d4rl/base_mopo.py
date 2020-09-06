from copy import deepcopy
from .base import base_params

mopo_params = deepcopy(base_params)
mopo_params['kwargs'].update({
    'sn': True,
    'separate_mean_var': True,
    'penalty_learned_var': True,
})