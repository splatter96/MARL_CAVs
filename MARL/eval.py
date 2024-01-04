import time
import gymnasium as gym

import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.append("../highway-env")
import highway_env
from sb3_contrib import SACD

import torch

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

def eval_policy(args):
    env = gym.make('merge-single-agent-v0')

    env.config["screen_height"] = 300
    env.config["screen_width"] = 2800
    env.config["safety_guarantee"] = False
    env.config["traffic_density"] = args.difficulty
    if args.mobil:
        env.config["action"] = {"type": "IDM"}
        env.config["action_masking"] = False

    if not args.mobil:
        model = SACD.load(args.model)

    num_tries = args.num_runs
    crashes = 0
    other_crashes = 0
    speed = 0
    road_speed = 0
    total_steps = 0

    # create the output directory if we need to save the trajectories
    if args.traj_dir != '' and not os.path.exists(args.traj_dir):
        os.mkdir(args.traj_dir)

    j = 0
    t = tqdm(range(num_tries))
    start = time.time()
    # import cProfile, pstats
    # profiler = cProfile.Profile()
    # profiler.enable()

    for i in t:
      done = truncated = False
      obs, info = env.reset()

      if args.initial_pos != '':
          load_veh = np.load(args.initial_pos, allow_pickle=True)

          env.road.vehicles = load_veh
          env.set_vehicle(env.road.vehicles[0])

      ret = 0
      position_list = []
      while not (done or truncated):
        if not args.mobil:
            action, _states = model.predict(obs, deterministic=True)
        else:
            action = None
        obs, reward, done, truncated, info = env.step(action)
        ret += reward
        # print(obs)

        speed += info["average_speed"]
        road_speed += info["average_road_speed"]
        total_steps += 1

        if args.traj_dir != '':
            veh_pos = info["vehicle_position"][0]
            position_list.append(veh_pos.copy())

        if args.render:
            env.render()
            time.sleep(0.1)

      if info["other_crashes"]:
          # print("crash happened")
          other_crashes += 1

      if info['crashed']:
          crashes += 1

      j += 1

      if info['crashed']:
      # if info["other_crashes"]:
          # only save trajectories if we didn't load any in the first place
          if args.initial_pos == '':
              np.save(f"initial_pos_{i}.npy", env.road.initial_vehicles)

      # if args.traj_dir != '':
          # np.save(f"{args.traj_dir}/pos_{i}.npy", np.array(position_list))

      # print(f"Episode done crashed:{info['crashed']}")
      # print(f"Current crashrate {crashes/(i+1)}")
      t.set_description(f"Crashrate {crashes/(i+1)} Other crashes {other_crashes/(i+1)}")

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats('eval_new.log')

    end = time.time()
    print(f"Took {(end-start)/j}")
    print(f"Crashrate {crashes/num_tries}")
    print(f"Average ego vehicle speed {speed/total_steps}")
    print(f"Average speed of all cars {road_speed/total_steps}")

if __name__ == "__main__":
    torch.set_num_threads(2)
    args = parse_args()
    eval_policy(args)

