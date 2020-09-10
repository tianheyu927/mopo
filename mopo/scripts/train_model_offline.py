import gym
import d4rl
import numpy as np
import tensorflow as tf

from mopo.models.constructor import construct_model, format_samples_for_training


def model_name(args):
    name = f'{args.env}-{args.quality}'
    if args.separate_mean_var:
        name += '_smv'
    name += f'_{args.seed}'
    return name


def main(args):
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    env = gym.make(f'{args.env}-{args.quality}-v0')
    dataset = env.get_dataset()
    obs_dim = dataset['observations'].shape[1]
    act_dim = dataset['actions'].shape[1]

    model = construct_model(obs_dim=obs_dim, act_dim=act_dim, hidden_dim=args.hidden_dim,
                            num_networks=args.num_networks, num_elites=args.num_elites,
                            model_type=args.model_type, separate_mean_var=args.separate_mean_var,
                            name=model_name(args))

    dataset['rewards'] = np.expand_dims(dataset['rewards'], 1)
    train_inputs, train_outputs = format_samples_for_training(dataset)
    model.train(train_inputs, train_outputs,
                batch_size=args.batch_size, holdout_ratio=args.holdout_ratio,
                max_epochs=args.max_epochs, max_t=args.max_t)
    model.save(args.model_dir, 0)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--env', required=True)
    parser.add_argument('--quality', required=True)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--model-type', default='mlp')
    parser.add_argument('--separate-mean-var', action='store_true')
    parser.add_argument('--num-networks', default=7, type=int)
    parser.add_argument('--num-elites', default=5, type=int)
    parser.add_argument('--hidden-dim', default=200, type=int)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--holdout-ratio', default=0.2, type=float)
    parser.add_argument('--max-epochs', default=None, type=int)
    parser.add_argument('--max-t', default=None, type=float)
    parser.add_argument('--model-dir', default='/tiger/u/gwthomas/d4rl/models')
    main(parser.parse_args())