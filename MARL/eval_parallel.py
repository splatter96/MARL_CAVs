import time
import gymnasium as gym

import pygame
import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

import matplotlib.cm as cm
import matplotlib as mpl

from PIL import Image

sys.path.append("../highway-env")
import highway_env
from sb3_contrib import SACD

import torch

import multiprocessing
from multiprocessing import Pool, Value


crash_pos = []
actions = []

env = None
model = None

t = None

def parse_args():
    parser = argparse.ArgumentParser(description=('Evaluate policy on RL environment'))
    parser.add_argument('model', nargs="?", type=str, help="Model which to evaluate")
    parser.add_argument('--difficulty', type=int, required=False,
                        default=3, help="difficulty setting to which the environment is to be set")
    parser.add_argument('--traj-dir', type=str, required=False,
                        default='', help="directory where to save the trajectories")
    parser.add_argument('--mobil', action='store_true', help="If set the mobil model instead of the RL agent")
    parser.add_argument('--num-runs', type=int, required=False,
                        default=200, help="number of runs to evaluate over")
    parser.add_argument('--initial-pos', type=str, required=False,
                        default='', help="numpy file with the initial positions to load")
    parser.add_argument( '--render', action='store_true',
        help='Wether to render the the output during evaluation or not')
    parser.add_argument( '--no-render', dest='render', action='store_false',
        help='Wether to render the the output during evaluation or not')
    parser.set_defaults(render=True)
    args = parser.parse_args()
    return args

def init(t_):
    ''' store the counter for later use '''
    global t

    t = t_

def eval_episode(num):
      global t

      local_speed = 0
      local_road_speed = 0
      local_steps = 0
      local_crashes = 0
      local_sucessfull_merges = 0

      done = truncated = False
      obs, info = env.reset()
      skip_run = False

      if args.initial_pos != '':
          load_veh = np.load(args.initial_pos, allow_pickle=True)

          env.road.vehicles = load_veh
          env.set_vehicle(env.road.vehicles[0])

      position_list = []
      while not (done or truncated):
        if not args.mobil:
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = None

        obs, reward, done, truncated, info = env.step(action)

        actions.append({'action': action, 'pos': info["vehicle_position"][0].copy()})
        # np.save("actions.npy", actions)

        local_speed += info["average_speed"]
        local_road_speed += info["average_road_speed"]
        local_steps += 1

        if args.traj_dir != '':
            veh_pos = info["vehicle_position"][0]
            position_list.append(veh_pos.copy())

        # also end the episode when another vehicle crashed
        if info["other_crashes"] and not info["crashed"]:
            skip_run = True

      # if info['crashed'] or info['other_crashes']:
          # if args.initial_pos == '':
              # np.save(f"initial_pos_{num}.npy", env.road.initial_vehicles)

      # if skip_run:
          # return

      if info['crashed']:
          local_crashes += 1
          #crash_pos.append(info["vehicle_position"][0])

      if info['merged']: # and not info["other_crashes"]:
          local_sucessfull_merges += 1

      # t.set_description(f"Crashrate {crashes.value/(i+1)} Mergerate {sucessfull_merges.value/(i+1)}")
      t.update()
      return (local_speed, local_road_speed, local_steps, local_crashes, local_sucessfull_merges)


def eval_policy(args):
    global env
    env = gym.make('merge-single-agent-v0')

    env.config["screen_height"] = 300
    env.config["screen_width"] = 2800
    env.config["safety_guarantee"] = False
    env.config["traffic_density"] = args.difficulty
    if args.mobil:
        env.config["action"] = {"type": "IDM"}
        env.config["action_masking"] = False

    global model
    if not args.mobil:
        model = SACD.load(args.model)

    # create the output directory if we need to save the trajectories
    if args.traj_dir != '' and not os.path.exists(args.traj_dir):
        os.mkdir(args.traj_dir)

    global t
    num_tries = args.num_runs
    t = tqdm(range(num_tries))
    start = time.time()

    pool =  multiprocessing.get_context('fork').Pool(processes=4, initializer = init, initargs = (t, ))

    crashes = 0
    other_crashes = 0
    speed = 0
    road_speed = 0
    steps = 0
    total_steps = 0
    sucessfull_merges = 0

    # global_speed = global_road_speed = global_steps = global_crashes = global_sucessfull_merges = 0
    for result in pool.map(eval_episode, range(num_tries), chunksize=50):
        local_speed, local_road_speed, local_steps, local_crashes, local_sucessfull_merges = result
        speed += local_speed
        road_speed += local_road_speed
        steps += local_steps
        crashes += local_crashes
        sucessfull_merges += local_sucessfull_merges

    # for i in t:
      # eval_episode(i)
      # t.set_description(f"Crashrate {crashes/(i+1)} Mergerate {sucessfull_merges/(i+1)}")

    end = time.time()
    print(f"Took {(end-start)}")
    print(f"Crashrate {crashes/num_tries}")
    print(f"Mergerate {sucessfull_merges/num_tries}")
    print(f"Average ego vehicle speed {speed/steps}")
    print(f"Average speed of all cars {road_speed/steps}")

    np.save("crash_pos.npy", crash_pos)
    np.save("actions.npy", actions)


if __name__ == "__main__":
    torch.set_num_threads(1)
    args = parse_args()
    eval_policy(args)

