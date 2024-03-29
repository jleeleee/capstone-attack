"""Deep Q learning graph

The functions in this file can are used to create the following functions:

======= act ========

    Function to chose an action given an observation

    Parameters
    ----------
    observation: object
        Observation that can be feed into the output of make_obs_ph
    stochastic: bool
        if set to False all the actions are always deterministic (default False)
    update_eps_ph: float
        update epsilon a new value, if negative not update happens
        (default: no update)

    Returns
    -------
    Tensor of dtype tf.int64 and shape (BATCH_SIZE,) with an action to be performed for
    every element of the batch.


======= train =======

    Function that takes a transition (s,a,r,s') and optimizes Bellman equation's error:

        td_error = Q(s,a) - (r + gamma * max_a' Q(s', a'))
        loss = huber_loss[td_error]

    Parameters
    ----------
    obs_t: object
        a batch of observations
    action: np.array
        actions that were selected upon seeing obs_t.
        dtype must be int32 and shape must be (batch_size,)
    reward: np.array
        immediate reward attained after executing those actions
        dtype must be float32 and shape must be (batch_size,)
    obs_tp1: object
        observations that followed obs_t
    done: np.array
        1 if obs_t was the last observation in the episode and 0 otherwise
        obs_tp1 gets ignored, but must be of the valid shape.
        dtype must be float32 and shape must be (batch_size,)
    weight: np.array
        imporance weights for every element of the batch (gradient is multiplied
        by the importance weight) dtype must be float32 and shape must be (batch_size,)

    Returns
    -------
    td_error: np.array
        a list of differences between Q(s,a) and the target in Bellman's equation.
        dtype is float32 and shape is (batch_size,)

======= update_target ========

    copy the parameters from optimized Q function to the target Q function.
    In Q learning we actually optimize the following error:

        Q(s,a) - (r + gamma * max_a' Q'(s', a'))

    Where Q' is lagging behind Q to stablize the learning. For example for Atari

    Q' is set to Q once every 10000 updates training steps.

"""
import tensorflow as tf
import rlattack.common.tf_util as U
from rlattack.str_attack import StrAttackL2
#V: Cleverhans imports#
from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, CarliniWagnerL2
from cleverhans.model import CallableModelWrapper
from cleverhans.compat import reduce_max, reduce_min
import numpy as np

from rlattack.deepq.map import SaliencyMap

#V: act function for test-time attacks/vanilla #
def build_act_enjoy (make_obs_ph, q_func, num_actions, noisy=False, scope="deepq", reuse=None, attack=None, model_path=''):
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values = q_func(observations_ph.get(), num_actions, scope="q_func", noisy=noisy)
        q_values = q_values.get_logits(observations_ph.get())
        #q_values = q_func(observations_ph, num_actions, scope="q_func", noisy=noisy)
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
        act2 = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=q_values,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
                         
        # Load model before attacks graph construction so that TF won't
        # complain can't load parameters for attack
        try:
            U.load_state(model_path)
        except:
            pass

        if attack != None:
            
            if attack == 'fgsm':
                def wrapper(x):
                    return q_func(x, num_actions, scope="q_func", reuse=True, concat_softmax=True, noisy=noisy)
                adversary = FastGradientMethod(CallableModelWrapper(wrapper, 'probs'), sess=U.get_session())
                adv_observations = adversary.generate(observations_ph.get(), eps=1.0/255.0,
                                                      clip_min=0, clip_max=1.0) * 255.0
            elif attack == 'iterative':
                def wrapper(x):
                    return q_func(x, num_actions, scope="q_func", reuse=True, concat_softmax=True)
                adversary = BasicIterativeMethod(CallableModelWrapper(wrapper, 'probs'), sess=U.get_session())
                adv_observations = adversary.generate(observations_ph.get(), eps=1.0/255.0,
                                                      clip_min=0, clip_max=1.0) * 255.0
            elif attack == 'cwl2':
                def wrapper(x):
                    return q_func(x, num_actions, scope="q_func", reuse=True)
                adversary = CarliniWagnerL2(CallableModelWrapper(wrapper, 'logits'), sess=U.get_session())
                cw_params = {'binary_search_steps': 1,
                             'max_iterations': 100,
                             'learning_rate': 0.1,
                             'initial_const': 10,
                             'clip_min': 0,
                             'clip_max': 1.0}
                adv_observations = adversary.generate(observations_ph.get(), **cw_params) * 255.0
            elif attack == 'strattack':
                def wrapper(x):
                    return q_func(x, num_actions, scope="q_func", reuse=True)
                adversary = StrAttackL2(CallableModelWrapper(wrapper, 'logits'), sess=U.get_session())
                str_params = {'binary_search_steps': 8,
                             'max_iterations': 100,
                             'learning_rate': 0.1,
                             'initial_const': 1}
                adv_observations = adversary.generate(observations_ph.get(), **str_params) * 255.0

            craft_adv_obs = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                            outputs=adv_observations,
                            givens={update_eps_ph: -1.0, stochastic_ph: True},
                            updates=[update_eps_expr])

        if attack == None:
            craft_adv_obs = None
            return act, act2
        else:
            return act, craft_adv_obs


def build_adv(make_obs_tf, q_func, num_actions, epsilon, noisy, attack='fgsm'):
    with tf.variable_scope('deepq', reuse=tf.AUTO_REUSE):
        obs_tf_in = U.ensure_tf_input(make_obs_tf("observation"))
        stochastic_ph_adv = tf.placeholder(tf.bool, (), name="stochastic_adv")
        update_eps_ph_adv = tf.placeholder(tf.float32, (), name="update_eps_adv")
        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        update_eps_expr_adv = eps.assign(tf.cond(update_eps_ph_adv >= 0, lambda: update_eps_ph_adv, lambda: eps))
        print ("==========================================")

        #def wrapper(x):
        #    return q_func(x, num_actions, scope="q_func", reuse=True, concat_softmax=True, noisy=noisy)
        #adversary = FastGradientMethod(q_func(obs_tf_in.get(), num_actions, scope="q_func", reuse=True, concat_softmax=True, noisy=noisy), sess=U.get_session())
        #adv_observations = adversary.generate(obs_tf_in.get(), eps=epsilon, clip_min=0, clip_max=1.0) * 255.0

        def print_target(gt):
            result = gt.copy()
            print("Attack target: " + str(np.argmax(result[0])))
            return result


        if attack == None or attack == 'fgsm':
            def wrapper(x):
                return q_func(x, num_actions, scope="q_func", reuse=True, concat_softmax=True, noisy=noisy)
            #adversary = FastGradientMethod(CallableModelWrapper(wrapper, 'probs'), sess=U.get_session())
            adversary = FastGradientMethod(q_func(obs_tf_in.get(), num_actions, scope="q_func", reuse=True, concat_softmax=True, noisy=noisy), sess=U.get_session())
            adv_observations = adversary.generate(obs_tf_in.get(), eps=epsilon,
                                                  clip_min=0, clip_max=1.0) * 255.0
        elif attack == 'iterative':
            def wrapper(x):
                return q_func(x, num_actions, scope="q_func", reuse=True, concat_softmax=True)
            #adversary = BasicIterativeMethod(CallableModelWrapper(wrapper, 'probs'), sess=U.get_session())
            adversary = BasicIterativeMethod(q_func(obs_tf_in.get(), num_actions, scope="q_func", reuse=True, concat_softmax=True, noisy=noisy), sess=U.get_session())
            adv_observations = adversary.generate(obs_tf_in.get(), eps=epsilon,
                                                  clip_min=0, clip_max=1.0) * 255.0
        elif attack == 'cwl2':
            def wrapper(x):
                return q_func(x, num_actions, scope="q_func", reuse=True)
            #adversary = CarliniWagnerL2(CallableModelWrapper(wrapper, 'logits'), sess=U.get_session())
            adversary = CarliniWagnerL2(q_func(obs_tf_in.get(), num_actions, scope="q_func", reuse=True, concat_softmax=True, noisy=noisy), sess=U.get_session())

            preds = adversary.model.get_probs(obs_tf_in.get())
            preds_max = reduce_min(preds, 1, keepdims=True)
            original_predictions = tf.to_float(tf.equal(preds, preds_max))
            labels = tf.stop_gradient(original_predictions)
            if isinstance(labels, np.ndarray):
              nb_classes = labels.shape[1]
            else:
              nb_classes = labels.get_shape().as_list()[1]

            y_target = tf.py_func(print_target, [labels],
                                       adversary.tf_dtype)
            y_target.set_shape([None, nb_classes])
            cw_params = {'binary_search_steps': 1,
                         'max_iterations': 100,
                         'learning_rate': 0.1,
                         'initial_const': 10,
                         'y_target': y_target,
                         'clip_min': 0,
                         'clip_max': 1.0}
            adv_observations = adversary.generate(obs_tf_in.get(), **cw_params) * 255.0
        elif attack == 'strattack':
            def wrapper(x):
                return q_func(x, num_actions, scope="q_func", reuse=True)
            #adversary = StrAttackL2(CallableModelWrapper(wrapper, 'logits'), sess=U.get_session())
            adversary = StrAttackL2(q_func(obs_tf_in.get(), num_actions, scope="q_func", reuse=True, concat_softmax=True, noisy=noisy), sess=U.get_session())

            preds = adversary.model.get_probs(obs_tf_in.get())
            preds_max = reduce_min(preds, 1, keepdims=True)
            original_predictions = tf.to_float(tf.equal(preds, preds_max))
            labels = tf.stop_gradient(original_predictions)
            if isinstance(labels, np.ndarray):
              nb_classes = labels.shape[1]
            else:
              nb_classes = labels.get_shape().as_list()[1]

            y_target = tf.py_func(print_target, [labels],
                                       adversary.tf_dtype)
            y_target.set_shape([None, nb_classes])

            str_params = {'binary_search_steps': 1,
                         'max_iterations': 100,
                         'learning_rate': 0.1,
                         'initial_const': 10,
                         'y_target': y_target,
                         'clip_min': 0,
                         'clip_max': 1.0,
                         'image_size': 84,
                         'stride': 2,
                         'filter_size': 2,
                         'num_channels': 1}

            adv_observations = adversary.generate(obs_tf_in.get(), **str_params) * 255.0
        else:
            print("Unknown attack specified")
            pass

        craft_adv_obs = U.function(inputs=[obs_tf_in, stochastic_ph_adv, update_eps_ph_adv],
                        outputs=adv_observations,
                        givens={update_eps_ph_adv: -1.0, stochastic_ph_adv: True},
                        updates=[update_eps_expr_adv])
        return craft_adv_obs

#######################

def build_act(make_obs_ph, q_func, num_actions, noisy=False, scope="deepq", reuse=None):
    """Creates the act function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that take a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    """
    with tf.variable_scope(scope, reuse=reuse):
        observations_ph = U.ensure_tf_input(make_obs_ph("observation"))
        stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
        update_eps_ph = tf.placeholder(tf.float32, (), name="update_eps")

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))

        q_values = q_func(observations_ph.get(), num_actions, scope="q_func", noisy=noisy)
        q_values = q_values.get_logits(observations_ph.get())
        deterministic_actions = tf.argmax(q_values, axis=1)

        batch_size = tf.shape(observations_ph.get())[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=num_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps
        stochastic_actions = tf.where(chose_random, random_actions, deterministic_actions)

        output_actions = tf.cond(stochastic_ph, lambda: stochastic_actions, lambda: deterministic_actions)
        update_eps_expr = eps.assign(tf.cond(update_eps_ph >= 0, lambda: update_eps_ph, lambda: eps))

        act = U.function(inputs=[observations_ph, stochastic_ph, update_eps_ph],
                         outputs=output_actions,
                         givens={update_eps_ph: -1.0, stochastic_ph: True},
                         updates=[update_eps_expr])
                         
        return act


def build_train(make_obs_ph, q_func, num_actions, optimizer, grad_norm_clipping=None, gamma=1.0, double_q=True, noisy=False, scope="deepq", reuse=None, attack=None):
    """Creates the train function:

    Parameters
    ----------
    make_obs_ph: str -> tf.placeholder or TfInput
        a function that takes a name and creates a placeholder of input with that name
    q_func: (tf.Variable, int, str, bool) -> tf.Variable
        the model that takes the following inputs:
            observation_in: object
                the output of observation placeholder
            num_actions: int
                number of actions
            scope: str
            reuse: bool
                should be passed to outer variable scope
        and returns a tensor of shape (batch_size, num_actions) with values of every action.
    num_actions: int
        number of actions
    reuse: bool
        whether or not to reuse the graph variables
    optimizer: tf.train.Optimizer
        optimizer to use for the Q-learning objective.
    grad_norm_clipping: float or None
        clip gradient norms to this value. If None no clipping is performed.
    gamma: float
        discount rate.
    double_q: bool
        if true will use Double Q Learning (https://arxiv.org/abs/1509.06461).
        In general it is a good idea to keep it enabled.
    scope: str or VariableScope
        optional scope for variable_scope.
    reuse: bool or None
        whether or not the variables should be reused. To be able to reuse the scope must be given.

    Returns
    -------
    act: (tf.Variable, bool, float) -> tf.Variable
        function to select and action given observation.
`       See the top of the file for details.
    train: (object, np.array, np.array, object, np.array, np.array) -> np.array
        optimize the error in Bellman's equation.
`       See the top of the file for details.
    update_target: () -> ()
        copy the parameters from optimized Q function to the target Q function.
`       See the top of the file for details.
    debug: {str: function}
        a bunch of functions to print debug data like q_values.
    """
    act_f = build_act(make_obs_ph, q_func, num_actions, scope=scope, noisy=noisy, reuse=reuse)

    with tf.variable_scope(scope, reuse=reuse):
        # set up placeholders
        obs_t_input = U.ensure_tf_input(make_obs_ph("obs_t"))
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        obs_tp1_input = U.ensure_tf_input(make_obs_ph("obs_tp1"))
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q network evaluation
        q_t = q_func(obs_t_input.get(), num_actions, scope="q_func", noisy=noisy, reuse=True)  # reuse parameters from act
        q_t = q_t.get_logits(obs_t_input.get())
        q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))

        # target q network evalution
        q_tp1 = q_func(obs_tp1_input.get(), num_actions, scope="target_q_func", noisy=noisy)
        q_tp1 = q_tp1.get_logits(obs_tp1_input.get())
        target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(q_t * tf.one_hot(act_t_ph, num_actions), 1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_using_online_net = q_func(obs_tp1_input.get(), num_actions, scope="q_func", noisy=noisy, reuse=True)
            q_tp1_using_online_net = q_tp1_using_online_net.get_logits(obs_tp1_input.get())
            q_tp1_best_using_online_net = tf.arg_max(q_tp1_using_online_net, 1)
            q_tp1_best = tf.reduce_sum(q_tp1 * tf.one_hot(q_tp1_best_using_online_net, num_actions), 1)
        else:
            q_tp1_best = tf.reduce_max(q_tp1, 1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = U.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)
        # compute optimization op (potentially with gradient clipping)
        if grad_norm_clipping is not None:
            optimize_expr = U.minimize_and_clip(optimizer,
                                                weighted_error,
                                                var_list=q_func_vars,
                                                clip_val=grad_norm_clipping)
        else:
            optimize_expr = optimizer.minimize(weighted_error, var_list=q_func_vars)

        # update_target_fn will be called periodically to copy Q network to target Q network
        update_target_expr = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_expr.append(var_target.assign(var))
        update_target_expr = tf.group(*update_target_expr)

        # Create callable functions
        train = U.function(
            inputs=[
                obs_t_input,
                act_t_ph,
                rew_t_ph,
                obs_tp1_input,
                done_mask_ph,
                importance_weights_ph
            ],
            outputs=[td_error, errors],
            updates=[optimize_expr]
        )
        update_target = U.function([], [], updates=[update_target_expr])

        q_values = U.function([obs_t_input], q_t)
        ################## Vahid's Work ###################
        #U.load_state(model_path)

        if attack != None:
            if attack == 'fgsm':
                #def wrapper(x):
                #    return q_func(x, num_actions, scope="target_q_func", reuse=True, concat_softmax=True, noisy=noisy)
                adversary = FastGradientMethod(q_func(obs_tp1_input.get(), num_actions, scope="target_q_func", reuse=True, concat_softmax=True, noisy=noisy), sess=U.get_session())
                adv_observations = adversary.generate(obs_tp1_input.get(), eps=1.0/255.0,
                                                      clip_min=0, clip_max=1.0) * 255.0
            elif attack == 'iterative':
                def wrapper(x):
                    return q_func(x, num_actions, scope="q_func", reuse=True, concat_softmax=True)
                adversary = BasicIterativeMethod(CallableModelWrapper(wrapper, 'probs'), sess=U.get_session())
                adv_observations = adversary.generate(observations_ph.get(), eps=1.0/255.0,
                                                      clip_min=0, clip_max=1.0) * 255.0
            elif attack == 'cwl2':
                def wrapper(x):
                    return q_func(x, num_actions, scope="q_func", reuse=True)
                adversary = CarliniWagnerL2(CallableModelWrapper(wrapper, 'logits'), sess=U.get_session())
                cw_params = {'binary_search_steps': 1,
                             'max_iterations': 100,
                             'learning_rate': 0.1,
                             'initial_const': 10,
                             'clip_min': 0,
                             'clip_max': 1.0}
                adv_observations = adversary.generate(observations_ph.get(), **cw_params) * 255.0
            elif attack == 'strattack':
                def wrapper(x):
                    return q_func(x, num_actions, scope="q_func", reuse=True)
                adversary = StrAttackL2(CallableModelWrapper(wrapper, 'logits'), sess=U.get_session())
                str_params = {'binary_search_steps': 1,
                             'max_iterations': 100,
                             'learning_rate': 0.1,
                             'initial_const': 10}
                adv_observations = adversary.generate(observations_ph.get(), **str_params) * 255.0

            craft_adv_obs = U.function(inputs=[obs_tp1_input],
                            outputs=adv_observations,                       
                            updates=[update_target_expr])

        if attack == None:
            craft_adv_obs = None

        return act_f, train, update_target, {'q_values': q_values}, craft_adv_obs



def build_map(make_obs_tf, q_func, num_actions, epsilon, noisy, output_shape):
    with tf.variable_scope('deepq', reuse=tf.AUTO_REUSE):
        obs_tf_in = U.ensure_tf_input(make_obs_tf("observation"))
        stochastic_ph_adv = tf.placeholder(tf.bool, (), name="stochastic_adv")
        update_eps_ph_adv = tf.placeholder(tf.float32, (), name="update_eps_adv")

        saliency_map = SaliencyMap(
            q_func(obs_tf_in.get(), num_actions, scope="q_func", reuse=True, concat_softmax=True, noisy=noisy),
            sess=U.get_session())

        eps = tf.get_variable("eps", (), initializer=tf.constant_initializer(0))
        update_eps_expr_adv = eps.assign(tf.cond(update_eps_ph_adv >= 0, lambda: update_eps_ph_adv, lambda: eps))
        print("==========================================")

        preds = saliency_map.model.get_probs(obs_tf_in.get())

        def print_true(gt):
            result = gt.copy()
            # print("Result: " + str(result[0]))
            return result

        preds_max = reduce_max(preds, 1, keepdims=True)
        true_predictions = tf.to_float(tf.equal(preds, preds_max))
        labels_true = tf.stop_gradient(true_predictions)
        if isinstance(labels_true, np.ndarray):
            nb_classes = labels_true.shape[1]
        else:
            nb_classes = labels_true.get_shape().as_list()[1]
        y_true = tf.py_func(print_true, [labels_true],
                              saliency_map.tf_dtype)
        y_true.set_shape([None, nb_classes])

        preds_min = reduce_min(preds, 1, keepdims=True)
        target_predictions = tf.to_float(tf.equal(preds, preds_min))
        labels_target = tf.stop_gradient(target_predictions)
        if isinstance(labels_target, np.ndarray):
            nb_classes = labels_target.shape[1]
        else:
            nb_classes = labels_target.get_shape().as_list()[1]
        y_target = tf.py_func(print_true, [labels_target],
                              saliency_map.tf_dtype)
        y_target.set_shape([None, nb_classes])

        sm_params = {'y_target': y_target,
                     'y_true': y_true,
                     }

        target_map = saliency_map.generate(obs_tf_in.get(), **sm_params)
        craft_map_obs = U.function(inputs=[obs_tf_in, stochastic_ph_adv, update_eps_ph_adv],
                                   outputs=target_map,
                                   givens={update_eps_ph_adv: -1.0, stochastic_ph_adv: True},
                                   updates=[update_eps_expr_adv])

        return craft_map_obs




