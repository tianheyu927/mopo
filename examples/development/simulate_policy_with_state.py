import argparse
from distutils.util import strtobool
import json
import os
import pickle

import numpy as np
import tensorflow as tf
import pdb

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
# from softlearning.samplers import rollouts
from softlearning import replay_pools
from softlearning.samplers import (
    dummy_sampler,
    extra_policy_info_sampler,
    remote_sampler,
    base_sampler,
    simple_sampler)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--num-rollouts', '-n', type=int, default=10)
    parser.add_argument('--render-mode', '-r',
                        type=str,
                        default=None,
                        choices=('human', 'rgb_array', None),
                        help="Mode to render the rollouts in.")
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

    args = parser.parse_args()

    return args

def rollout(env,
            policy,
            path_length,
            callback=None,
            render_mode=None,
            break_on_terminal=True):
    observation_space = env.observation_space
    action_space = env.action_space

    pool = replay_pools.SimpleReplayPool(
        observation_space, action_space, max_size=path_length)
    sampler = simple_sampler.SimpleSampler(
        max_path_length=path_length,
        min_pool_size=None,
        batch_size=None)

    sampler.initialize(env, policy, pool)
    images = []
    infos = []
    state_vectors = []

    t = 0
    for t in range(path_length):
        observation, reward, terminal, info = sampler.sample()
        state_vector = sampler.env.unwrapped.state_vector()
        infos.append(info)
        state_vectors.append(state_vector)

        if callback is not None:
            callback(observation)

        if render_mode is not None:
            if render_mode == 'rgb_array':
                image = env.render(mode=render_mode)
                images.append(image)
            else:
                env.render()

        if terminal:
            policy.reset()
            if break_on_terminal: break

    assert pool._size == t + 1

    path = pool.batch_by_indices(
        np.arange(pool._size),
        observation_keys=getattr(env, 'observation_keys', None))
    path['infos'] = infos
    path['state_vectors'] = np.array([sampler._reset_state_vector] + state_vectors[:-1])
    if render_mode == 'rgb_array':
        path['images'] = np.stack(images, axis=0)

    return path


def rollouts(n_paths, *args, **kwargs):
    paths = [rollout(*args, **kwargs) for i in range(n_paths)]
    return paths

def simulate_policy(args):
    session = tf.keras.backend.get_session()
    checkpoint_path = args.checkpoint_path.rstrip('/')
    experiment_path = os.path.dirname(checkpoint_path)

    variant_path = os.path.join(experiment_path, 'params.json')
    with open(variant_path, 'r') as f:
        variant = json.load(f)

    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])
    evaluation_environment = get_environment_from_params(environment_params)

    policy = (
        get_policy_from_variant(variant, evaluation_environment, Qs=[None]))
    policy.set_weights(picklable['policy_weights'])

    with policy.set_deterministic(args.deterministic):
        paths = rollouts(args.num_rollouts,
                         evaluation_environment,
                         policy,
                         path_length=args.max_path_length,
                         render_mode=args.render_mode)

    #### print rewards
    rewards = [path['rewards'].sum() for path in paths]
    print('Rewards: {}'.format(rewards))
    print('Mean: {}'.format(np.mean(rewards)))
    ####
    
    if args.render_mode != 'human':
        from pprint import pprint; import pdb; pdb.set_trace()
        pass

    return paths


if __name__ == '__main__':
    args = parse_args()
    paths = simulate_policy(args)

    keys = paths[0].keys()
    paths = {key: np.concatenate([path[key] for path in paths]) for key in keys}

    print(paths.keys())
    print(paths['observations'].shape, paths['state_vectors'].shape)
    # pickle.dump(paths, open('data/hopper_state_vectors.pkl', 'wb'))


