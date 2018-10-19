import os

import gym
import retro

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, ActorCriticPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.cmd_util import make_atari_env, make_atari_snapshot_env
from stable_baselines.common.vec_env import VecFrameStack

from stable_baselines import PPO2, ACER


ENV_NAME = "BreakoutNoFrameskip-v4"

NUM_CPU = 12
# env = make_atari_env(ENV_NAME, num_env=NUM_CPU , seed=37)
# # Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=4)
# Create and wrap the environmen
# env = SubprocVecEnv([lambda: gym.make(ENV_NAME) for i in range(NUM_CPU)])

env = retro.make(game='SuperMarioWorld-Snes')
env = DummyVecEnv([lambda: env])

# Add some param noise for exploration
model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=f"./Tensorboard/PPO2_{ENV_NAME}/",
            gamma=0.99,
            lam=.95,
            vf_coef=1,
            ent_coef=0.005,
            noptepochs=3,
            cliprange=0.1,
            learning_rate=5e-4,
            nminibatches=4,
            n_steps=32)

# model = ACER(CnnPolicy, env, verbose=1, tensorboard_log=f"./Tensorboard/ACER_PongNoFrameskip-v4/")

# Train the agent
#model = model.load(f"SavedModels/PPO2_{ENV_NAME}", env=env, tensorboard_log=f"./Tensorboard/PPO2_{ENV_NAME}/")
model.learn(total_timesteps= 8000000)
model.save(f"SavedModels/PPO2_{ENV_NAME}")


# Enjoy trained agent
obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
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
