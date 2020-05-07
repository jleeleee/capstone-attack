from cleverhans.attacks import SaliencyMapMethod
import warnings

import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.compat import reduce_sum, reduce_max, reduce_any

tf_dtype = tf.as_dtype('float32')

class SaliencyMapOnly(SaliencyMapMethod):
    """
    Just the relevant parts for creating the Saliency Map
    """
    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param kwargs: See `parse_params`
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        # Create random targets if y_target not provided
        if self.y_target is None:
            from random import randint

            def random_targets(gt):
                result = gt.copy()
                nb_s = gt.shape[0]
                nb_classes = gt.shape[1]

                for i in range(nb_s):
                    result[i, :] = np.roll(result[i, :],
                                           randint(1, nb_classes - 1))

                return result

            labels, nb_classes = self.get_or_guess_labels(x, kwargs)
            self.y_target = tf.py_func(random_targets, [labels],
                                       self.tf_dtype)
            self.y_target.set_shape([None, nb_classes])

        x_adv = asm_symbolic(
            x,
            model=self.model,
            y_target=self.y_target,
            theta=self.theta,
            gamma=self.gamma,
            clip_min=self.clip_min,
            clip_max=self.clip_max)
        return x_adv


def asm_symbolic(x, y_target, model, theta, gamma, clip_min, clip_max):
    """
    TensorFlow implementation of the JSMA (see https://arxiv.org/abs/1511.07528
    for details about the algorithm design choices).
    :param x: the input placeholder
    :param y_target: the target tensor
    :param model: a cleverhans.model.Model object.
    :param theta: delta for each feature adjustment
    :param gamma: a float between 0 - 1 indicating the maximum distortion
        percentage
    :param clip_min: minimum value for components of the example returned
    :param clip_max: maximum value for components of the example returned
    :return: a tensor for the adversarial saliency map
    """

    nb_classes = int(y_target.shape[-1].value)
    nb_features = int(np.product(x.shape[1:]).value)

    if x.dtype == tf.float32 and y_target.dtype == tf.int64:
        y_target = tf.cast(y_target, tf.int32)

    if x.dtype == tf.float32 and y_target.dtype == tf.float64:
        warnings.warn("Downcasting labels---this should be harmless unless"
                      " they are smoothed")
        y_target = tf.cast(y_target, tf.float32)

    logits = model.get_logits(x)
    # create the Jacobian graph
    list_derivatives = []
    for class_ind in xrange(nb_classes):
        derivatives = tf.gradients(logits[:, class_ind], x)
        list_derivatives.append(derivatives[0])
    grads = tf.reshape(
        tf.stack(list_derivatives), shape=[nb_classes, -1, nb_features])

    # Compute the Jacobian components
    # To help with the computation later, reshape the target_class
    # and other_class to [nb_classes, -1, 1].
    # The last dimention is added to allow broadcasting later.
    target_class = tf.reshape(
        tf.transpose(y_target, perm=[1, 0]), shape=[nb_classes, -1, 1])
    other_classes = tf.cast(tf.not_equal(target_class, 1), tf_dtype)

    grads_target = reduce_sum(grads * target_class, axis=0)
    grads_other = reduce_sum(grads * other_classes, axis=0)

    print(grads_target)
    return grads
