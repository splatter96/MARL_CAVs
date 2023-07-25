import sys
sys.path.append("../highway-env")
import highway_env
import time
import gym

#import gymnasium as gym
# from matplotlib import pyplot as plt

# env = gym.make('merge-v1')
env = gym.make('merge-single-agent')

env.config["screen_height"] = 300
env.config["screen_width"] = 1000
env.config["safety_guarantee"] = False

env.reset()
done  = False
while not done:
    action = env.action_type.actions_indexes["SLOWER"]
    obs, reward, done, info = env.step(action)
    # print(obs)
    time.sleep(0.1)
    env.render()
