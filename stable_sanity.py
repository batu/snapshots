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
import gym_minigrid
from gym_minigrid.wrappers import FlatObsWrapper
from stable_baselines import PPO2, DQN, PPO2_SH, ACER

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multiprocessing training (num_env=4 => 4 processes)

env =  gym.make("MiniGrid-Empty-6x6-v0")
env = FlatObsWrapper(env)
env = DummyVecEnv([lambda: env])

# env = SnapshotVecEnv([lambda: gym.make("MiniGrid-Empty-6x6-v0")],
#                        snapshot_save_prob=0,
#                        snapshot_load_prob=0,
#                        human_snapshots=0,)


best_mean_reward, n_steps = -np.inf, 0
last_name = "dummy"
def callback(_locals, _globals):
  """
  Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward, last_name
  # Print stats every 1000 calls

  if (n_steps + 1) % 100 == 0:
      # Evaluate policy performance
      x, y = ts2xy(load_results(run_path), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(f"Save probability: {save_prob}, load probability: {load_prob}")
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              if os.path.exists(run_path + f'/{last_name}'):
                  os.remove(run_path + f'/{last_name}')
              print("Saving new best model")
              last_name = f"{best_mean_reward:.1f}_{ENV_NAME}_model.pkl"
              _locals['self'].save(run_path + f'/{best_mean_reward:.1f}_{ENV_NAME}_model.pkl')
  n_steps += 1
  return False

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000, callback=callback)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
