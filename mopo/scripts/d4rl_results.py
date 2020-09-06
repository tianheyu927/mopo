from collections import defaultdict, namedtuple
from glob import glob
import json
from itertools import product
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ENVS = ['halfcheetah', 'hopper', 'walker2d']
QUALITIES = ['random', 'medium', 'mixed', 'medium_expert']
IDENTIFYING_KEYS = ['rollout_length', 'penalty_coeff', 'penalty_learned_var', 'separate_mean_var', 'sn']

MIN_ITERS = 50
LAST_K = 10

D4RL_BASELINES = {
    'halfcheetah': (
        (12135.0,  -17.9, 3502.0, 2885.6, 2641.0, 3207.3),
        (12135.0, 4196.4, -808.6, 4508.7, 5178.2, 5365.3),
        (12135.0, 4492.1, -581.3, 4211.3, 5384.7, 5413.8),
        (12135.0, 4169.4,  -55.7, 6132.5, 5156.0, 5342.4)
    ),
    'hopper': (
        (3234.3,  299.4, 347.7,  289.5, 341.0,  370.5),
        (3234.3,  923.5,   5.7, 1527.9, 994.8, 1030.0),
        (3234.3,  364.4,  93.3,  802.7,   2.0,    5.3),
        (3234.3, 3621.2,  32.9,  109.8,  16.0,    5.1)
    ),
    'walker2d': (
        (4592.3,  73.0, 192.0,  307.6,  38.4,    23.9),
        (4592.3, 304.8,  44.2, 1526.7, 3341.1, 3734.3),
        (4592.3, 518.6,  87.8,  495.3,  -11.5,   44.5),
        (4592.3, 297.0,  -5.1, 1193.6,  141.7, 3058.0)
    )
}
BASELINE_ALGS = ('SAC', 'BC', 'SAC-off', 'BEAR', 'BRAC-p', 'BRAC-v')
D4RL_BASELINES = {
    env: {
        quality: {
            alg: value for alg, value in zip(BASELINE_ALGS, row)
        }
        for quality, row in zip(QUALITIES, data)
    }
    for env, data in D4RL_BASELINES.items()
}
D4RL_BASELINES = {f'{env}_{quality}': D4RL_BASELINES[env][quality] for env, quality in product(ENVS, QUALITIES)}

import pdb

def name(variant):
    h = variant['rollout_length']
    coeff = variant['penalty_coeff']
    s = f'h={h}, coeff={coeff}'
    if variant['sn']:
        s += ', sn'
    if variant['separate_mean_var']:
        s += ', smv'
    if variant['penalty_learned_var']:
        s += ', plv'
    return s

def skip(variant):
    return 'deterministic' in variant and variant['deterministic'] == True

def get_results(args):
    dir = Path(args.dir)
    all_results = {}
    for env, quality in product(ENVS, QUALITIES):
        domain = f'{env}_{quality}'
        all_results[domain] = defaultdict(list)
        for subdir in (dir/env).iterdir():
            if not subdir.name.startswith(domain):
                continue
            if quality == 'medium' and subdir.name.startswith(f'{env}_medium_expert'):
                continue

            seed_dirs = glob(f'{subdir}/seed:*')
            for seed_dir in seed_dirs:
                seed_dir = Path(seed_dir)

                params_path = seed_dir/'params.json'
                if not params_path.is_file():
                    print(f'No params; skipping')
                    continue
                params = json.loads(params_path.read_text())
                seed = params['run_params']['seed']
                alg_params = params['algorithm_params']['kwargs']
                if skip(alg_params):
                    continue
                identifier = name({
                    key: alg_params.get(key, None) for key in IDENTIFYING_KEYS
                })

                results_path = seed_dir/'result.json'
                if not results_path.is_file():
                    print(f'No results for seed {seed}; skipping')
                    continue
                results = [json.loads(line)['evaluation/return-average'] for line in results_path.read_text().splitlines()]
                if len(results) < MIN_ITERS:
                    print(f'Fewer than {MIN_ITERS} results for seed {seed}; skipping')
                    continue
                all_results[domain][identifier].append(results)
    return all_results

def main(args):
    all_results = get_results(args)
    for domain, results in all_results.items():
        print(domain)
        for id, result in results.items():
            last_results = [np.mean(r[-LAST_K:]) for r in result]
            best = np.argmax(last_results)
            print('\t', id, ':', last_results[best])
            all_results[domain][id] = result[best]

    Path(args.out_dir).mkdir(exist_ok=True)
    for domain, results in all_results.items():
        del D4RL_BASELINES[domain]['SAC-off']
        plt.figure()
        for baseline, value in D4RL_BASELINES[domain].items():
            plt.plot(np.ones(1000) * value, label=baseline, linestyle='--')
        for id, result in results.items():
            plt.plot(result, label=id)
        plt.legend()
        plt.savefig(f'{args.out_dir}/{domain}', bbox_inches='tight')

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', default='/tiger/u/gwthomas/ray_mopo')
    parser.add_argument('--out-dir', default='/tiger/u/gwthomas/d4rl/plots')
    parser.parse_args()
    main(parser.parse_args())