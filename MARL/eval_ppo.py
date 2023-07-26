import time
import gymnasium as gym

import sys
import numpy as np
import warnings
warnings.filterwarnings('ignore')


sys.path.append("../highway-env")
import highway_env
from stable_baselines3 import PPO

# config = {
            # "action": {
              # "type": "DiscreteMetaAction",
              # "target_speeds": np.linspace(10, 30, 5).tolist()
            # },
            # "observation": {
              # "type": "Kinematics",
              # "see_behind": True,
              # "clip": False,
            # }
         # }

# env = gym.make('Foo-v0', render_mode='rgb_array', config=config)
env = gym.make('merge-single-agent-v0')

env.config["screen_height"] = 300
env.config["screen_width"] = 1900
#env.config["action"]["target_speeds"] = np.linspace(10, 30, 3).tolist()
env.config["safety_guarantee"] = False
env.config["traffic_density"] = 1


# Load and test saved model
#model = PPO.load("highway_ppo/model")
# model = PPO.load("/home/paul/model_new")
# model = PPO.load("/home/paul/model_lower_speed")
#model = PPO.load("/home/paul/model_lower_speed_bigger_arch_lower_lc_cost.zip")
# model = PPO.load("/home/paul/model_parallel_env.zip")
#model = PPO.load("/home/paul/model_parallel_env_higher_cost.zip")
# model = PPO.load("/home/paul/model_parallel_env_all_spawns.zip")
# model = PPO.load("/home/paul/model_parallel_env_all_spawns_lower_speed.zip")
#model = PPO.load("/home/paul/model_parallel_env_all_spawns_marl_params_see_behind.zip")
# model = PPO.load("/home/paul/model_parallel_env_merge_spawn_marl_params_see_behind_no_clip.zip")
# model = PPO.load("/home/paul/model_first_test_shorter.zip")

#model = PPO.load("/home/paul/model_my_road.zip")
model = PPO.load("/home/paul/model_my_road_continued_density_2.zip")
# model = PPO.load("/home/paul/model_default_params.zip")

num_tries = 100
crashes = 0
# while True:
for i in range(num_tries):
  done = truncated = False
  obs, info = env.reset()
  ret = 0
  while not (done or truncated):
    action, _states = model.predict(obs, deterministic=True)
    # print(f"{action}")
    # print(f"{info=}")
    obs, reward, done, truncated, info = env.step(action)
    # print(obs[0])
    # print(obs[1])
    # print("\n")
    ret += reward

    # env.render()
    # time.sleep(0.05)

  if info['crashed']:
      crashes += 1

  print(f"Episode done crashed:{info['crashed']} return: {ret}")

print(f"Crashrate {crashes/num_tries}")
