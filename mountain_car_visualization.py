import os
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, ActorCriticPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, SnapshotVecEnv
from stable_baselines.common.cmd_util import make_atari_env, make_atari_snapshot_env
from stable_baselines.common.vec_env import VecFrameStack

from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.deepq.policies import MlpPolicy
from stable_baselines import DQN

from stable_baselines import PPO2, DQN, PPO2_SH

ENV_NAME = "MountainCar-v0"
NUM_CPU = 1


env = gym.make('MountainCar-v0')
env = SnapshotVecEnv([lambda: gym.make(ENV_NAME)], snapshot_save_prob=0.001, snapshot_load_prob=0.75)

model = DQN(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=400000)

env = DummyVecEnv([lambda: gym.make(ENV_NAME)])

# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        env.reset()
    env.render()


#     :param policy: (ActorCriticPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, CnnLstmPolicy, ...)
#     :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
#     :param gamma: (float) Discount factor
#     :param n_steps: (int) The number of steps to run for each environment per update
#         (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
#     :param ent_coef: (float) Entropy coefficient for the loss caculation
#     :param learning_rate: (float or callable) The learning rate, it can be a function
#     :param vf_coef: (float) Value function coefficient for the loss calculation
#     :param max_grad_norm: (float) The maximum value for the gradient clipping
#     :param lam: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator
#     :param nminibatches: (int) Number of training minibatches per update. For recurrent policies,
#         the number of environments run in parallel should be a multiple of nminibatches.
#     :param noptepochs: (int) Number of epoch when optimizing the surrogate
#     :param cliprange: (float or callable) Clipping parameter, it can be a function
#     :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
#     :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
#     :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
