import time
# import gym
import gymnasium as gym
import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')


sys.path.append("../highway-env")
import highway_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv

if __name__ == "__main__":
    # config = {
                # "action": {
                  # "type": "DiscreteMetaAction",
                  # "target_speeds": np.linspace(10, 30, 3).tolist()
                # },
                # "observation": {
                  # "type": "Kinematics",
                  # "see_behind": True
                # }
                # "safety_guarantee" : False
             # }

    # env = make_vec_env('merge-single-agent', n_envs=8, vec_env_cls=SubprocVecEnv, env_kwargs={"config": config})
    # env = make_vec_env('merge-single-agent-v0', n_envs=8, vec_env_cls=SubprocVecEnv)
    env = gym.make('merge-single-agent-v0')
    env.config["safety_guarantee"] = False

    model = PPO('MlpPolicy', env,
                  policy_kwargs=dict(net_arch=[256, 256]),
                  learning_rate=5e-4,
                  #n_steps=1000000,
                  #n_steps=4096,
                  batch_size=256,
                  gamma=0.99,
                  verbose=1,
                  #tensorboard_log="highway_ppo/",
                  device='cpu')
    model.learn(int(10e5))
    model.save("model_first_test.zip")
