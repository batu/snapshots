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

from stable_baselines import PPO2, DQN, PPO2_SH
# BREADCRUMBS_START
NUM_CPU = 12
ENV_NAME = "BreakoutNoFrameskip-v4"
# BREADCRUMBS_END
save_prob, load_prob, experiment_name = parseArguments()


REPLAY = False
TB_path = f"Results/Tensorboard/{experiment_name}/"

run_number = 0
try:
    os.mkdir(TB_path[:-1])
except:
    pass

try:
    os.mkdir(f"{TB_path}README")
except:
    pass

models_path = "Results/SavedModels/"

changes = """ Changed to the new and improved RL library. Started using PPO"""
reasoning = """PPO is the go to algorithm."""
hypothesis = """Better results. Not intially but after tweaking. """


if not REPLAY:
    if len(hypothesis) + len(changes) + len(reasoning) < 10:
        print("NOT ENOUGH LOGGING INFO")
        print("Please write more about the changes and reasoning.")
        exit()

    with open(f"{TB_path}/README/README.txt", "w") as readme:
        start_time_ascii = time.asctime(time.localtime(time.time()))
        algorithm = os.path.basename(__file__)[:-2]
        print(f"Experiment start time: {start_time_ascii}", file=readme)
        print(f"\nAlgorithm:\n{algorithm}", file=readme)
        print(f"\nThe Changes:\n{changes}", file=readme)
        print(f"\nReasoning:\n{reasoning}", file=readme)
        print(f"\nHypothesis:\n{hypothesis}", file=readme)
        print(f"\nResults:\n", file=readme)


folder_count = len([f for f in os.listdir(TB_path) if not os.path.isfile(os.path.join(models_path, f))])
run_name = f"run{folder_count}"
run_path = f'{TB_path}{run_name}'
os.mkdir(run_path)

# This function saves all the important hypterparameters to the run summary file.
save_hyperparameters(["experiment.py"], f"{run_path}/run_summary.txt", save_prob=save_prob, load_prob=load_prob, experiment_name=experiment_name)



def make_env(rank):
    def _thunk():
        env = make_atari(ENV_NAME)
        env.seed(37 + rank)
        env = Monitor(env, filename=run_path, allow_early_resets=True)
        return wrap_deepmind(env)
    return _thunk
set_global_seeds(37)


start_time_ascii = time.asctime(time.localtime(time.time()))
start_time = time.time()
print("Training has started!")

# BREADCRUMBS_START
# Create an OpenAIgym environment.
#env = make_atari_snapshot_env(ENV_NAME, num_env=NUM_CPU , seed=37, snapshot_save_prob=0.001, snapshot_load_prob=0.75)

env = SnapshotVecEnv([make_env(i) for i in range(NUM_CPU)],
                       snapshot_save_prob=0.001,
                       snapshot_load_prob=1)

# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

# Add some param noise for exploration
model = PPO2(CnnPolicy, env, verbose=1, tensorboard_log=f"{run_path}",
             gamma=0.99,
             lam=.95,
             vf_coef=1,
             ent_coef=0.005,
             noptepochs=3,
             cliprange=0.1,
             learning_rate=5e-4,
             nminibatches=1,
             n_steps=32,
             )

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

  if (n_steps + 1) % 10 == 0:
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

# Train the agent
model.learn(total_timesteps= 40000000, callback=callback)
# BREADCRUMBS_END
model.save(f"{run_path}/{ENV_NAME}_final.pkl")

print("The training has completed!")




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
