from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np

def get_required_argument(dotmap, key, message, default=None):
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val

def spectral_norm(w, iteration=1):
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

   u_hat = u
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = tf.nn.l2_normalize(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = tf.nn.l2_normalize(u_)

   u_hat = tf.stop_gradient(u_hat)
   v_hat = tf.stop_gradient(v_hat)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = w / sigma
       w_norm = tf.reshape(w_norm, w_shape)


   return w_norm

class TensorStandardScaler:
    """Helper class for automatically normalizing inputs into the network.
    """
    def __init__(self, x_dim):
        """Initializes a scaler.

        Arguments:
        x_dim (int): The dimensionality of the inputs into the scaler.

        Returns: None.
        """
        self.fitted = False
        with tf.variable_scope("Scaler"):
            self.mu = tf.get_variable(
                name="scaler_mu", shape=[1, x_dim], initializer=tf.constant_initializer(0.0),
                trainable=False
            )
            self.sigma = tf.get_variable(
                name="scaler_std", shape=[1, x_dim], initializer=tf.constant_initializer(1.0),
                trainable=False
            )

        self.cached_mu, self.cached_sigma = np.zeros([0, x_dim]), np.ones([1, x_dim])

    def fit(self, data):
        """Runs two ops, one for assigning the mean of the data to the internal mean, and
        another for assigning the standard deviation of the data to the internal standard deviation.
        This function must be called within a 'with <session>.as_default()' block.

        Arguments:
        data (np.ndarray): A numpy array containing the input

        Returns: None.
        """
        mu = np.mean(data, axis=0, keepdims=True)
        sigma = np.std(data, axis=0, keepdims=True)
        sigma[sigma < 1e-12] = 1.0

        self.mu.load(mu)
        self.sigma.load(sigma)
        self.fitted = True
        self.cache()

    def transform(self, data):
        """Transforms the input matrix data using the parameters of this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return (data - self.mu) / self.sigma

    def inverse_transform(self, data):
        """Undoes the transformation performed by this scaler.

        Arguments:
        data (np.array): A numpy array containing the points to be transformed.

        Returns: (np.array) The transformed dataset.
        """
        return self.sigma * data + self.mu

    def get_vars(self):
        """Returns a list of variables managed by this object.

        Returns: (list<tf.Variable>) The list of variables.
        """
        return [self.mu, self.sigma]

    def cache(self):
        """Caches current values of this scaler.

        Returns: None.
        """
        self.cached_mu = self.mu.eval()
        self.cached_sigma = self.sigma.eval()

    def load_cache(self):
        """Loads values from the cache

        Returns: None.
        """
        self.mu.load(self.cached_mu)
        self.sigma.load(self.cached_sigma)

