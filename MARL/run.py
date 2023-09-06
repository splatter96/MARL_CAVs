import sys
sys.path.append("../highway-env")
import highway_env
import time

import warnings
warnings.filterwarnings('ignore')

import gymnasium as gym
# from matplotlib import pyplot as plt

# env = gym.make('merge-v1')
env = gym.make('merge-single-agent')

env.config["screen_height"] = 300
env.config["screen_width"] = 1000
env.config["safety_guarantee"] = False
env.config["traffic_density"] = 3

env.reset()
done  = False
action = env.action_type.actions_indexes["SLOWER"]

#warmup step
for _ in range(5):
    env.step(action)


# import cProfile, pstats
# profiler = cProfile.Profile()
# profiler.enable()

start = time.time()
for _ in range(1000):
    env.step(action)
    # obs, reward, done, info = env.step(action)
    # print(obs)
    time.sleep(0.1)
    env.render()

end = time.time()
print(f"Took {(end-start)/1000 * 1000}")

# profiler.disable()
# stats = pstats.Stats(profiler)
# stats.dump_stats('myProfile_opt1.log')
