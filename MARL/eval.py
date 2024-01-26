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

last_action_prob = None

def display_action(action_surface, sim_surface):
    cell_size = [action_surface.get_width()/5, 300]

    action_map = [
        'LANE_LEFT',
        'IDLE',
        'LANE_RIGHT',
        'FASTER',
        'SLOWER'
        ]

    global last_action_prob
    if last_action_prob is None:
        return

    for i, value in np.ndenumerate(last_action_prob):
        i = i[1]
        cmap = cm.plasma
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        color = cmap(norm(value), bytes=True)
        pygame.draw.rect(action_surface, color, (cell_size[0]*i, 0, cell_size[0], cell_size[1]), 0)

        font = pygame.font.Font(None, 15)
        # probilities
        text = "p={:.2f}".format(value)
        text = font.render(text,
                          1, (10, 10, 10), (255, 255, 255))
        action_surface.blit(text, (cell_size[0]*i, 0))

        #action text
        text = f"{action_map[i]}"
        text = font.render(text,
                          1, (10, 10, 10), (255, 255, 255))
        action_surface.blit(text, (cell_size[0]*i, 20))

def eval_policy(args):
    # env = gym.make('merge-single-agent-v0', render_mode='rgb_array')
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
    sucessfull_merges = 0

    # create the output directory if we need to save the trajectories
    if args.traj_dir != '' and not os.path.exists(args.traj_dir):
        os.mkdir(args.traj_dir)

    j = 0
    t = tqdm(range(num_tries))
    start = time.time()

    for i in t:
      done = truncated = False
      obs, info = env.reset()
      #set the envviewr in the env
      env.render()
      env.viewer.set_agent_display(display_action)
      skip_run = False

      if args.initial_pos != '':
          load_veh = np.load(args.initial_pos, allow_pickle=True)

          env.road.vehicles = load_veh
          env.set_vehicle(env.road.vehicles[0])

      ret = 0
      position_list = []
      while not (done or truncated):
        if not args.mobil:
            action, _states = model.predict(obs, deterministic=True)
            t_obs = torch.tensor(obs)
            t_obs = t_obs[None, :]
            action_prob, action_log_prob = model.policy.actor.action_log_prob(t_obs)
            global last_action_prob
            last_action_prob = action_prob.detach().numpy()
        else:
            action = None
        obs, reward, done, truncated, info = env.step(action)
        ret += reward

        speed += info["average_speed"]
        road_speed += info["average_road_speed"]
        total_steps += 1

        if args.traj_dir != '':
            veh_pos = info["vehicle_position"][0]
            position_list.append(veh_pos.copy())

        if args.render:
            env.render()
            time.sleep(0.1)

        # also end the episode when another vehicle crashed
        #if info["other_crashes"] and not info["crashed"]:
            # frame = env.render()
            # im = Image.fromarray(frame)
            # im.save(f"crash_{i}.png")
            #skip_run = True
            #print("other crash")

      #if info['crashed']:
      # if info["other_crashes"]:
      if skip_run:
          # only save trajectories if we didn't load any in the first place
          if args.initial_pos == '':
              np.save(f"initial_pos_{i}.npy", env.road.initial_vehicles)

      if skip_run:
        continue

      #if info["other_crashes"]:
          #other_crashes += 1

      if info['crashed']:
          crashes += 1

      #if info['merged']: # and not info["other_crashes"]:
          #sucessfull_merges += 1

      j += 1

      # if args.traj_dir != '':
          # np.save(f"{args.traj_dir}/pos_{i}.npy", np.array(position_list))

      # print(f"Episode done crashed:{info['crashed']}")
      # print(f"Current crashrate {crashes/(i+1)}")
      # t.set_description(f"Crashrate {crashes/(i+1)} Other crashes {other_crashes/(i+1)}")
      # t.set_description(f"Crashrate {crashes/(i+1)} Mergerate {sucessfull_merges/(i+1-other_crashes)}")
      t.set_description(f"Crashrate {crashes/(i+1)} Mergerate {sucessfull_merges/(i+1)}")

    end = time.time()
    print(f"Took {(end-start)/j}")
    print(f"Crashrate {crashes/num_tries}")
    print(f"Average ego vehicle speed {speed/total_steps}")
    print(f"Average speed of all cars {road_speed/total_steps}")

if __name__ == "__main__":
    torch.set_num_threads(2)
    args = parse_args()
    eval_policy(args)

