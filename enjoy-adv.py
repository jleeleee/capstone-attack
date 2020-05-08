""" DQN - Test-time attacks

============ Sample usage ============ 

No attack, testing a DQN model of Breakout trained without parameter noise:
$> python3 enjoy-adv.py --env Breakout --model-dir ./data/Breakout/model-173000 --video ./Breakout.mp4

No attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation):
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --video ./Breakout.mp4

Whitebox FGSM attack, testing a DQN model of Breakout trained without parameter noise:
$> python3 enjoy-adv.py --env Breakout --model-dir ./data/Breakout/model-173000 --attack fgsm --video ./Breakout.mp4

Whitebox FGSM attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation):
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --attack fgsm --video ./Breakout.mp4

Blackbox FGSM attack, testing a DQN model of Breakout trained without parameter noise:
$> python3 enjoy-adv.py --env Breakout --model-dir ./data/Breakout/model-173000 --attack fgsm --blackbox --model-dir2 ./data/Breakout/model-173000-2 --video ./Breakout.mp4

Blackbox FGSM attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation), replica model trained without parameter noise:
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --attack fgsm --blackbox --model-dir2 ./data/Breakout/model-173000-2 --video ./Breakout.mp4

Blackbox FGSM attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation), replica model trained with parameter noise:
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --attack fgsm --blackbox --model-dir2 ./data/Breakout/model-173000-2 --noisy2 --video ./Breakout.mp4

"""

import argparse
import gym
import os
import numpy as np
from scipy.special import softmax

#from gym.monitoring import VideoRecorder
from gym import wrappers
from time import time
import rlattack.common.tf_util as U

from PIL import Image

from rlattack import deepq
from rlattack.common.misc_util import (
        boolean_flag,
        SimpleMonitor,
        )
from rlattack.common.atari_wrappers_deprecated import wrap_dqn
#from rlattack.deepq.experiments.atari.model import model, dueling_model

#V: imports#
import tensorflow as tf
import cv2
from collections import deque
from model import model, dueling_model
from statistics import statistics

class DQNModel:
        """
        Creating Q-graph, FGSM graph
        Supports loading multiple graphs - needed for blackbox attacks
        """

        def __init__(self, env, dueling, noisy, fname, attack=None):
            self.g = tf.Graph()
            self.noisy = noisy
            self.dueling = dueling
            self.env = env
            with self.g.as_default():
                self.act = deepq.build_act_enjoy(
                        make_obs_ph=lambda name: U.Uint8Input(env.observation_space.shape, name=name),
                        q_func=dueling_model if dueling else model,
                        num_actions=env.action_space.n,
                        noisy=noisy
                        )
                self.q_val = self.act[1]
                self.act = self.act[0]
                self.saver = tf.train.Saver()
            self.sess = tf.Session(graph=self.g)
            self.attack = attack

            if fname is not None:
                print ('Loading Model...')
                self.saver.restore(self.sess, fname)

        def get_act(self):
                return self.act

        def get_session(self):
                return self.sess 

        def craft_adv(self):
            with self.sess.as_default():
                with self.g.as_default():
                    craft_adv_obs = deepq.build_adv(
                        make_obs_tf=lambda name: U.Uint8Input(self.env.observation_space.shape, name=name),
                        q_func=dueling_model if self.dueling else model,
                        num_actions=self.env.action_space.n,
                        epsilon = 1.0/255.0,
                        noisy=self.noisy,
                        attack=self.attack
                        )
            return craft_adv_obs

        def craft_map(self):
            with self.sess.as_default():
                with self.g.as_default():
                    craft_map = deepq.build_map(
                        make_obs_tf=lambda name: U.Uint8Input(self.env.observation_space.shape, name=name),
                        q_func=dueling_model if self.dueling else model,
                        num_actions=self.env.action_space.n,
                        epsilon=1.0 / 255.0,
                        noisy=self.noisy,
                        output_shape=(None, self.env.action_space.n)
                    )
            return craft_map



def parse_args():
    parser = argparse.ArgumentParser("Run an already learned DQN model.")
    # Environment
    parser.add_argument("--env", type=str, required=True, help="name of the game")
    parser.add_argument("--model-dir", type=str, default=None, help="load model from this directory. ")
    parser.add_argument("--video", type=str, default=None, help="Path to mp4 file where the video of first episode will be recorded.")
    boolean_flag(parser, "stochastic", default=True, help="whether or not to use stochastic actions according to models eps value")
    boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
    #V: Attack Arguments#
    parser.add_argument("--model-dir2", type=str, default=None, help="load adversarial model from this directory (blackbox attacks). ")
    parser.add_argument("--attack", type=str, default=None, help="Method to attack the model.")
    boolean_flag(parser, "noisy", default=False, help="whether or not to NoisyNetwork")
    boolean_flag(parser, "noisy2", default=False, help="whether or not to NoisyNetwork")
    boolean_flag(parser, "blackbox", default=False, help="whether or not to NoisyNetwork")

    return parser.parse_args()


def make_env(game_name):
    env = gym.make(game_name + "NoFrameskip-v4")
    #env = SimpleMonitor(env)
    env = wrappers.Monitor(env, './videos/' + str(time()) + '/')
    env = wrap_dqn(env)
    return env

def img_stats(img, show=False):
    print("Image max: " + str(np.max(np.max(img, axis=1), axis=1)))
    print("Image min: " + str(np.min(np.min(img, axis=1), axis=1)))
    if show:
        for i in range(4):
            image = Image.fromarray(np.floor(img[0,:,:,i]).astype(np.uint8), 'P')
            print(img[0,:,:,i])
            print(np.floor(img[0,:,:,i]).astype(np.uint8))
            image.show()



def perturbation_stats(adv, scale=1.0):
    tmp = adv/scale
    l2 = np.linalg.norm(tmp, axis=(1,2))
    print("Perturbation l2: " + str(l2))
    print("Perturbation max: " + str(np.max(np.max(tmp, axis=1), axis=1)))
    print("Perturbation min: " + str(np.min(np.min(tmp, axis=1), axis=1)))

def is_score(asm, perturbation, nu=90):
    B_asm = asm.copy()
    thresh = np.percentile(B_asm, nu)
    print(thresh)
    B_asm[B_asm < thresh] = 0
    B_asm[B_asm >= thresh] = 1
    B_asm_perturb = B_asm * perturbation
    numer = np.linalg.norm(B_asm_perturb)
    denom = np.linalg.norm(perturbation)
    return numer/denom, B_asm_perturb.reshape((1, 84, 84, 4))

def play(env, act, craft_adv_obs, craft_adv_obs2, stochastic, video_path, attack, m_target, m_adv, craft_asm):
        num_episodes = 0
        num_moves = 0
        num_attacks = 0
        num_transfer = 0
        episode_rewards = [0.0]
        #video_recorder = None
        #video_recorder = VideoRecorder(
        #	env, video_path, enabled=video_path is not None)
        obs = env.reset()
        is_record = []
        while True:
                env.unwrapped.render()
                #video_recorder.capture_frame()
                num_moves += 1

                #V: Attack #
                if attack != None:
                        # Craft adv. examples
                        # STRATEGICALLY TIMED ATTACK
                        q_vals = softmax(m_target.q_val(np.array(obs)[None], stochastic=stochastic)[0])
                        max_q = max(q_vals)
                        min_q = min(q_vals)
                        diff = max_q - min_q
                        thresh = 0
                        if diff < thresh:
                            action = act(np.array(obs)[None], stochastic=stochastic)[0]

                        else:
                            num_attacks += 1
                            with m_adv.get_session().as_default():
                                adv_obs = craft_adv_obs(np.array(obs)[None], stochastic_adv=stochastic)[0]
                            with m_target.get_session().as_default():
                                action = act(np.array(adv_obs)[None], stochastic=stochastic)[0]
                                action2 = act(np.array(obs)[None], stochastic=stochastic)[0]
                                if (action != action2):
                                    print("Attacked: changed {} to {} with q_vals={}".format(action2, action, q_vals))
                                    num_transfer += 1

                                np_adv = np.array(adv_obs)[None]
                                np_obs = np.array(obs)[None]
                                adv_perturbation = np_adv - np_obs
                                print(adv_perturbation)
                                print("Original:")
                                # img_stats(np_obs, True)
                                print("Adversarial:")
                                # img_stats(np_adv, True)
                                # img_stats(adv_perturbation,True)
                                perturbation_stats(adv_perturbation)
                                print(">")
                                print("************************\n CREATING MAP \n ")
                                adv_map = craft_asm(
                                    np.array(obs)[None],
                                    stochastic_adv=stochastic,
                                )
                                scaled_img_map = adv_map.copy().reshape((1,84,84,4))
                                for frame in range(4):
                                    max_val = np.max(scaled_img_map[:,:,:,frame])
                                    img = scaled_img_map[:,:,:,frame]/max_val
                                    scaled_img_map[:,:,:,frame] = img*255
                                interpretability, perturb_map = is_score(adv_map, adv_perturbation.reshape((84*84*4)), nu=99)
                                print("Interpretability Score: {}".format(interpretability))
                                is_record.append(interpretability)
                                # img_stats(perturb_map, True)
                                print("************************")
                                # quit()
                else:
                        # Normal
                        action = act(np.array(obs)[None], stochastic=stochastic)[0]

                obs, rew, done, _ = env.step(action)
                episode_rewards[-1] += rew
                if done:
                        obs = env.reset()
                        episode_rewards.append(0.0)


                if done:
                        #if len(info["rewards"]) == 1: #and video_recorder.enabled:
                                # save video of first episode
                                #print("Saved video.")
                                #video_recorder.close()
                                #video_recorder.enabled = False
                        #mean_100ep_reward = round(np.mean(episode_rewards[-101:-1]), 1)
                        print('Reward: ' + str(episode_rewards[-2]))
                        num_episodes = len(episode_rewards)
                        print ('Episode: ' + str(num_episodes))
                        if num_moves > 0:
                            rate = float((num_attacks)/num_moves) * 100.0
                            print("Percentage of moves attacked: "+str(rate))
                        if num_attacks > 0:
                            success = float((num_transfer)/num_attacks) * 100.0
                            print("Percentage of successful attacks: "+str(success))
                        num_moves = 0
                        num_transfer = 0
                        num_attacks = 0
                        if len(is_record) > 0:
                            print("Avg interp: {}".format(np.mean(is_record)))
                            print(is_record)
                        return


if __name__ == '__main__':
    args = parse_args()
    env = make_env(args.env)
    g1 = tf.Graph()
    g2 = tf.Graph()
    with g1.as_default():
        m1 = DQNModel(env, args.dueling, args.noisy, os.path.join(args.model_dir, "saved"), args.attack)
    if args.blackbox == True:
        with g2.as_default():
            m2 = DQNModel(env, args.dueling, args.noisy2, os.path.join(args.model_dir2, "saved"))
            with m2.get_session().as_default():
                craft_adv_obs = m2.craft_adv()
            with m1.get_session().as_default():
                craft_adv_obs2 = m1.craft_adv()
                play(env, m1.get_act(), craft_adv_obs, craft_adv_obs2, args.stochastic, args.video, args.attack, m1, m2)
    else:
        with m1.get_session().as_default():
            craft_asm = m1.craft_map()
            craft_adv_obs = m1.craft_adv()
            play(env, m1.get_act(), craft_adv_obs, None, args.stochastic, args.video, args.attack, m1, m1, craft_asm)
