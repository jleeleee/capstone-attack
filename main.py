import gym
import random

from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.atari_wrappers import make_atari
from stable_baselines import DQN

def main():
    env = make_atari('BreakoutNoFrameskip-v4')
    env.reset()

    model = DQN(CnnPolicy, env, verbose=1)
    print(model.params[0].name)
    model.params[0].name = model.params[0].name[:-2]
    print(model.params[0].name)
    #tf.saved_model.simple_save(model.sess, "testpoint.ckpt", inputs={"obs": model.act_model.obs_ph},
                                   #outputs={"action": model.act_model._policy_proba})
    exit()
    model.learn(total_timesteps=10000)

    episode_reward = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        episode_reward += reward
        if done:
            print('Reward: %s' % episode_reward)
            break

if __name__ == '__main__':
    main()

