import time
import gymnasium as gym

import pygame
import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import matplotlib.cm as cm
import matplotlib as mpl

sys.path.remove("/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env")
sys.path.remove("/home/paul/Documents/PhD/RL/highway_env_commonroad/highway-env")

sys.path.append("../highway-env")
import highway_env
from sb3_contrib import SACD

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=("Evaluate policy on RL environment"))
    parser.add_argument("model", nargs="?", type=str, help="Model which to evaluate")
    parser.add_argument(
        "--difficulty",
        type=int,
        required=False,
        default=3,
        help="difficulty setting to which the environment is to be set",
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        required=False,
        default="",
        help="directory where to save the trajectories",
    )
    parser.add_argument(
        "--mobil",
        action="store_true",
        help="If set the mobil model instead of the RL agent",
    )
    parser.add_argument(
        "--merging",
        action="store_true",
        help="If set use the merging instead of the weaving environment",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        required=False,
        default=200,
        help="number of runs to evaluate over",
    )
    parser.add_argument(
        "--initial-pos",
        type=str,
        required=False,
        default="",
        help="numpy file with the initial positions to load",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Wether to render the the output during evaluation or not",
    )
    parser.add_argument(
        "--no-render",
        dest="render",
        action="store_false",
        help="Wether to render the the output during evaluation or not",
    )
    parser.set_defaults(render=True)
    args = parser.parse_args()
    return args


last_action_prob = None
last_observation = None
last_info = None


def display_action(action_surface, sim_surface):
    def angle_to_position(angle, _range):
        x = np.cos(angle) * _range
        y = np.sin(angle) * _range

        return np.array([x, y])

    global last_observation
    # TODO get these parameters from the actual agent
    cells = 16
    maximum_range = 150
    angle = 2 * np.pi / cells

    lidar_color = (255, 0, 0)

    if (
        last_info is None
        or last_observation is None
        or not "vehicle_position" in last_info
    ):
        return

    ranges = last_observation[:, 0]

    for i, _range in enumerate(ranges):
        pos = angle_to_position(angle * i, _range * maximum_range)
        pos += last_info["vehicle_position"][0]

        pos = sim_surface.pos2pix(pos[0], pos[1])
        # TODO account for heading of the vehicle
        pygame.draw.rect(sim_surface, lidar_color, (pos[0], pos[1], 10, 10))

    cell_size = [action_surface.get_width() / 5, 300]

    action_map = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

    global last_action_prob
    if last_action_prob is None:
        return

    for i, value in np.ndenumerate(last_action_prob):
        i = i[1]
        cmap = cm.plasma
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        color = cmap(norm(value), bytes=True)
        pygame.draw.rect(
            action_surface, color, (cell_size[0] * i, 0, cell_size[0], cell_size[1]), 0
        )

        font = pygame.font.Font(None, 15)
        # probilities
        text = "p={:.2f}".format(value)
        text = font.render(text, 1, (10, 10, 10), (255, 255, 255))
        action_surface.blit(text, (cell_size[0] * i, 0))

        # action text
        text = f"{action_map[i]}"
        text = font.render(text, 1, (10, 10, 10), (255, 255, 255))
        action_surface.blit(text, (cell_size[0] * i, 20))


def eval_policy(args):
    env = gym.make("merge-single-agent-v0")

    config = {}
    config["screen_height"] = 300
    config["screen_width"] = 2800
    config["safety_guarantee"] = False
    config["traffic_density"] = args.difficulty

    if args.merging:
        config["use_weaving"] = False

    if args.mobil:
        config["action"] = {"type": "IDM"}
        config["action_masking"] = False

    env = gym.make("merge-single-agent-v0", config=config)

    if not args.mobil:
        model = SACD.load(args.model)
        # model.set_random_seed(21)
        model.set_random_seed(32)

    # Number of successful crashes we aim for
    target_crashes = args.num_runs

    crashes = 0
    other_crashes = 0
    speed = 0
    road_speed = 0
    total_steps = 0
    sucessfull_merges = 0

    # create the output directory if we need to save the trajectories
    if args.traj_dir != "" and not os.path.exists(args.traj_dir):
        os.mkdir(args.traj_dir)

    j = 0  # episode counter
    # Use tqdm without a predefined total; we update it each episode.
    t = tqdm(desc="Evaluation episodes", total=target_crashes)
    start = time.time()

    action_map = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]

    # Run episodes until we have observed ``target_crashes`` crashes.
    # while crashes < target_crashes:
    while j < target_crashes:
        done = truncated = False
        obs, info = env.reset()
        # set the envviewr in the env
        if args.render:
            env.render()
            # env.viewer.set_agent_display(display_action)
        skip_run = False

        if args.initial_pos != "":
            load_veh = np.load(args.initial_pos, allow_pickle=True)

            env.road.vehicles = load_veh
            env.set_vehicle(env.road.vehicles[0])

        # ret = 0
        # position_list = []
        # action_buffer = []
        # initialise per‑episode storage for trajectory positions if needed
        position_list = []
        while not (done or truncated):
            if not args.mobil:
                action, _states = model.predict(obs, deterministic=True)
                # print(action_map[action])
                # t_obs = torch.tensor(obs)
                # t_obs = t_obs[None, :]
                # action_prob, action_log_prob = model.policy.actor.action_log_prob(t_obs)
                # global last_action_prob
                # last_action_prob = action_prob.detach().numpy()
                # action_buffer.append(last_action_prob[0])
            else:
                action = None
            obs, reward, done, truncated, info = env.step(action)
            # ret += reward

            # global last_observation
            # last_observation = obs
            # global last_info
            # last_info = info

            speed += info["average_speed"]
            road_speed += info["average_road_speed"]
            total_steps += 1

            if args.traj_dir != "":
                veh_pos = info["vehicle_position"][0]
                position_list.append(veh_pos.copy())

            if args.render:
                env.render()
                time.sleep(0.1)

            # also end the episode when another vehicle crashed
            # if info["other_crashes"] and not info["crashed"]:
                # skip_run = True
                # break

            # print(info["rear_end_collision_by_other"])
            # if info["rear_end_collision_by_other"]:
            #     break

            if info["merged"]:
                break

        # if skip_run:
        #     continue

        if info["other_crashes"] and not info["crashed"]:
            other_crashes += 1
            # print("other have crashed!!!!!\n\n")

        if info["crashed"]: # and not info["rear_end_collision_by_other"]:
            # t.update(1)
            crashes += 1
            # only save trajectories if we didn't load any in the first place
            if args.initial_pos == "":
                np.save(f"initial_pos_{j}.npy", env.road.initial_vehicles)
            # np.save(f"action_before_crash_{j}.npy", action_buffer)
        # else:
        # np.save(f"action_without_crash_{j}.npy", action_buffer)

        if "merged" in info and info["merged"]:  # and not info["other_crashes"]:
            sucessfull_merges += 1

        # if not info["rear_end_collision_by_other"]:
        #     j += 1

            
        j += 1
        t.update(1)

        # Update the progress description with current statistics.
        # Guard against division by zero – j is always >=1 here.
        t.set_description(
            f"Crashrate {crashes/j:.3f} Mergerate {sucessfull_merges/j:.3f} other_crashes {other_crashes/j:.3f}"
        )

    end = time.time()
    print(f"Took {(end-start)/j:.3f} seconds per episode (average)")
    print(f"Crashrate {crashes/j:.3f}")
    print(f"Average ego vehicle speed {speed/total_steps:.3f}")
    print(f"Average speed of all cars {road_speed/total_steps:.3f}")


if __name__ == "__main__":
    torch.set_num_threads(2)
    args = parse_args()

    eval_policy(args)
