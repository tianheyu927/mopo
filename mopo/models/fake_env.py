import numpy as np
import tensorflow as tf
import pdb

class FakeEnv:
    def __init__(self, model, config,
                penalty_coeff=0.,
                penalty_learned_var=False,
                penalty_learned_var_random=False):
        self.model = model
        self.config = config
        self.penalty_coeff = penalty_coeff
        self.penalty_learned_var = penalty_learned_var
        self.penalty_learned_var_random = penalty_learned_var_random

    '''
        x : [ batch_size, obs_dim + 1 ]
        means : [ num_models, batch_size, obs_dim + 1 ]
        vars : [ num_models, batch_size, obs_dim + 1 ]
    '''
    def _get_logprob(self, x, means, variances):

        k = x.shape[-1]

        ## [ num_networks, batch_size ]
        log_prob = -1/2 * (k * np.log(2*np.pi) + np.log(variances).sum(-1) + (np.power(x-means, 2)/variances).sum(-1))
        
        ## [ batch_size ]
        prob = np.exp(log_prob).sum(0)

        ## [ batch_size ]
        log_prob = np.log(prob)

        stds = np.std(means,0).mean(-1)

        return log_prob, stds

    def step(self, obs, act, deterministic=False):
        assert len(obs.shape) == len(act.shape)
        if len(obs.shape) == 1:
            obs = obs[None]
            act = act[None]
            return_single = True
        else:
            return_single = False

        inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means[:,:,1:] += obs
        ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds

        if not deterministic:
            #### choose one model from ensemble
            num_models, batch_size, _ = ensemble_model_means.shape
            model_inds = self.model.random_inds(batch_size)
            batch_inds = np.arange(0, batch_size)
            samples = ensemble_samples[model_inds, batch_inds]
            model_means = ensemble_model_means[model_inds, batch_inds]
            model_stds = ensemble_model_stds[model_inds, batch_inds]
            ####
        else:
            samples = np.mean(ensemble_samples, axis=0)
            model_means = np.mean(ensemble_model_means, axis=0)
            model_stds = np.mean(ensemble_model_stds, axis=0)

        log_prob, dev = self._get_logprob(samples, ensemble_model_means, ensemble_model_vars)

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_fn(obs, act, next_obs)

        batch_size = model_means.shape[0]
        return_means = np.concatenate((model_means[:,:1], terminals, model_means[:,1:]), axis=-1)
        return_stds = np.concatenate((model_stds[:,:1], np.zeros((batch_size,1)), model_stds[:,1:]), axis=-1)

        if self.penalty_coeff != 0:
            if not self.penalty_learned_var:
                ensemble_means_obs = ensemble_model_means[:,:,1:]
                mean_obs_means = np.mean(ensemble_means_obs, axis=0)     # average predictions over models
                diffs = ensemble_means_obs - mean_obs_means
                normalize_diffs = False
                if normalize_diffs:
                    obs_dim = next_obs.shape[1]
                    obs_sigma = self.model.scaler.cached_sigma[0,:obs_dim]
                    diffs = diffs / obs_sigma
                dists = np.linalg.norm(diffs, axis=2)   # distance in obs space
                penalty = np.max(dists, axis=0)         # max distances over models
            else:
                penalty = np.amax(np.linalg.norm(ensemble_model_stds, axis=2), axis=0)

            penalty = np.expand_dims(penalty, 1)
            assert penalty.shape == rewards.shape
            unpenalized_rewards = rewards
            penalized_rewards = rewards - self.penalty_coeff * penalty
        else:
            penalty = None
            unpenalized_rewards = rewards
            penalized_rewards = rewards

        if return_single:
            next_obs = next_obs[0]
            return_means = return_means[0]
            return_stds = return_stds[0]
            unpenalized_rewards = unpenalized_rewards[0]
            penalized_rewards = penalized_rewards[0]
            terminals = terminals[0]

        info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev,
                'unpenalized_rewards': unpenalized_rewards, 'penalty': penalty, 'penalized_rewards': penalized_rewards}
        return next_obs, penalized_rewards, terminals, info

    ## for debugging computation graph
    def step_ph(self, obs_ph, act_ph, deterministic=False):
        assert len(obs_ph.shape) == len(act_ph.shape)

        inputs = tf.concat([obs_ph, act_ph], axis=1)
        # inputs = np.concatenate((obs, act), axis=-1)
        ensemble_model_means, ensemble_model_vars = self.model.create_prediction_tensors(inputs, factored=True)
        # ensemble_model_means, ensemble_model_vars = self.model.predict(inputs, factored=True)
        ensemble_model_means = tf.concat([ensemble_model_means[:,:,0:1], ensemble_model_means[:,:,1:] + obs_ph[None]], axis=-1)
        # ensemble_model_means[:,:,1:] += obs_ph
        ensemble_model_stds = tf.sqrt(ensemble_model_vars)
        # ensemble_model_stds = np.sqrt(ensemble_model_vars)

        if deterministic:
            ensemble_samples = ensemble_model_means
        else:
            # ensemble_samples = ensemble_model_means + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
            ensemble_samples = ensemble_model_means + tf.random.normal(tf.shape(ensemble_model_means)) * ensemble_model_stds

        samples = ensemble_samples[0]

        rewards, next_obs = samples[:,:1], samples[:,1:]
        terminals = self.config.termination_ph_fn(obs_ph, act_ph, next_obs)
        info = {}

        return next_obs, rewards, terminals, info

    def close(self):
        pass



