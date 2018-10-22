import os
import gym
import time
import sys
import argparse
from support_utils import save_hyperparameters, parseArguments

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines import logger
from stable_baselines.common.policies import MlpPolicy, FeedForwardPolicy, ActorCriticPolicy, CnnPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv, SnapshotVecEnv
from stable_baselines.common.cmd_util import make_atari_env, make_atari_snapshot_env
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines.bench import Monitor
from stable_baselines.common import set_global_seeds
from stable_baselines.results_plotter import load_results, ts2xy

from stable_baselines import PPO2, DQN, PPO2_SH, ACER

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multiprocessing training (num_env=4 => 4 processes)
env = make_atari_snapshot_env('PongNoFrameskip-v4', num_env=4, seed=0, snapshot_save_prob=0.00, snapshot_load_prob=0.0)
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = ACER(CnnPolicy, env, verbose=1)
model.learn(total_timesteps=25000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
