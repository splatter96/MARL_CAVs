import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import time

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("../highway-env")
import highway_env

from Discrete_SAC_Agent import SACAgent

TRAINING_EVALUATION_RATIO = 4
EPISODES_PER_RUN = 100

if __name__ == "__main__":
    env = gym.make('merge-single-agent-v0')
    env.config["safety_guarantee"] = False
    env.config["screen_height"] = 300
    env.config["screen_width"] = 1900

    agent = SACAgent(env)
    agent.load('/home/paul/', 0, train_mode=True)
    crashes  = 0
    for episode_number in range(EPISODES_PER_RUN):
        episode_reward = 0
        state, _ = env.reset()
        state = state.flatten()
        done = False

        while not done:
            action = agent.get_next_action(state, evaluation_episode=True)

            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.flatten()

            env.render()
            time.sleep(0.1)

            state = next_state

        if info['crashed']:
          crashes += 1

        print(f"Crashed {info['crashed']}")

    print(f"Crashrate {crashes/EPISODES_PER_RUN}")

