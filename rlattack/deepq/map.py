from cleverhans.attacks import SaliencyMapMethod
import warnings

import numpy as np
from six.moves import xrange
import tensorflow as tf

from cleverhans.attacks.attack import Attack
from cleverhans.compat import reduce_sum, reduce_max, reduce_any

from random import randint

tf_dtype = tf.as_dtype('float32')

class SaliencyMapOnly(SaliencyMapMethod):
    """
    Just the relevant parts for creating the Saliency Map
    """

    def __init__(self, model, sess=None, dtypestr='float32', **kwargs):
        print(**kwargs)
        super(SaliencyMapOnly, self).__init__(model, sess, dtypestr, **kwargs)
        self.feedable_kwargs = ('y_target', 'y_true')

    def generate(self, x, **kwargs):
        """
        Generate symbolic graph for adversarial examples and return.
        :param x: The model's symbolic inputs.
        :param kwargs: See `parse_params`
        """
        # Parse and save attack-specific parameters
        assert self.parse_params(**kwargs)

        x_adv = asm_symbolic(
            x,
            model=self.model,
            y_target=self.y_target,
            y_true=self.y_true,
        )
        return x_adv

    def parse_params(self,
                     y_target=None,
                     y_true=None,
                     symbolic_impl=True,
                     **kwargs):
        self.y_target = y_target
        self.y_true = y_true
        return True

def scale_to_range(tensor, scaler=1):
    tensor = tf.div(
        tf.subtract(
            tensor,
            tf.reduce_min(tensor)
        ),
        tf.subtract(
            tf.reduce_max(tensor),
            tf.reduce_min(tensor)
        )
    )
    return tensor*scaler

def asm_symbolic(x, y_target, y_true, model):
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

    if x.dtype == tf.float32 and y_true.dtype == tf.int64:
        y_true = tf.cast(y_true, tf.int32)

    if x.dtype == tf.float32 and y_true.dtype == tf.float64:
        warnings.warn("Downcasting labels---this should be harmless unless"
                      " they are smoothed")
        y_true = tf.cast(y_true, tf.float32)

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
    true_class = tf.reshape(
        tf.transpose(y_true, perm=[1, 0]), shape=[nb_classes, -1, 1])

    grads_target = reduce_sum(grads * target_class, axis=0)
    grads_true = reduce_sum(grads * true_class, axis=0)

    z_grads_target = tf.nn.relu(grads_target)
    z_grads_true = tf.nn.relu(tf.negative(grads_true))
    asm = tf.multiply(z_grads_true, z_grads_target)

    # print(grads_target)
    return asm
