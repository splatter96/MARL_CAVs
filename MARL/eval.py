import time
import gymnasium as gym
import functools

import pygame
import argparse
import sys
import os
import numpy as np
from tqdm import tqdm
import warnings
import random

warnings.filterwarnings("ignore")

import matplotlib.cm as cm
import matplotlib as mpl

from PIL import Image

if "/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env" in sys.path:
    sys.path.remove("/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env")
sys.path.append("../highway-env")

import highway_env
from highway_env.utils import lmap

from sb3_contrib import SACD

import torch

import cProfile, pstats


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


def display_vehicles(action_surface, sim_surface, env):
    # if env.unwrapped.paused:
    #     return

    obs = env.unwrapped.observation_type.observe()
    obs = obs.reshape(5, -1)
    obs_type = env.observation_type
    color = (255, 255, 255)
    for v_index in range(obs.shape[0]):
        v_position = {}
        for feature in ["x", "y"]:
            v_feature = obs[v_index, obs_type.features.index(feature)]
            v_feature = lmap(v_feature, [-1, 1], obs_type.features_range[feature])
            v_position[feature] = v_feature

        # TODO
        # overwrite the x position according to lane distance measure
        # lane = env.unwrapped.vehicle.lane
        # ego_abs = env.unwrapped.vehicle.position
        # ego_rel_x = lane.local_coordinates(ego_abs)[0]
        # absolut_x = lane.position(v_position["x"] + ego_rel_x, 0)[0]
        # v_position["x"] = absolut_x
        #
        v_position = np.array([v_position["x"], v_position["y"]])
        if not obs_type.absolute and v_index > 0:
            v_position += env.unwrapped.vehicle.position
            # v_position[1] += env.unwrapped.vehicle.position[1]

        # print(f"{v_index} {v_position}")
        if v_index > 0 and obs[v_index, 0] == 1:
            pygame.draw.line(
                sim_surface,
                color,
                sim_surface.vec2pix(env.vehicle.position),
                sim_surface.vec2pix(v_position),
                7,
            )


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

    # ranges = last_observation[:, 0]
    #
    # for i, _range in enumerate(ranges):
    #     pos = angle_to_position(angle * i, _range * maximum_range)
    #     pos += last_info["vehicle_position"][0]
    #
    #     pos = sim_surface.pos2pix(pos[0], pos[1])
    #     # TODO account for heading of the vehicle
    #     pygame.draw.rect(sim_surface, lidar_color, (pos[0], pos[1], 10, 10))

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
    ## EVAL SEED
    ## This seed is used for
    ## all the evalution runs to be comparable
    seed_ = 22

    # set reproduceable seed
    random.seed(seed_)
    torch.random.manual_seed(seed_)
    np.random.seed(seed_)

    # env = gym.make('merge-single-agent-v0', render_mode='rgb_array')
    env = gym.make("merge-single-agent-v0")

    # set reproduceable seed for the env
    env.action_space.seed(seed_)
    env.reset(seed=seed_)

    env.config["screen_height"] = 300
    env.config["screen_width"] = 2800
    # env.config["screen_height"] = 1920
    # env.config["screen_width"] = 1920
    env.config["safety_guarantee"] = False
    env.config["traffic_density"] = args.difficulty
    # env.config["policy_frequency"] = 125
    # env.config["simulation_frequency"] = 125
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
    if args.traj_dir != "" and not os.path.exists(args.traj_dir):
        os.mkdir(args.traj_dir)

    j = 0
    t = tqdm(range(num_tries))
    start = time.time()

    # profiler = cProfile.Profile()
    # profiler.enable()
    #
    # import cProfile, pstats

    # profiler = cProfile.Profile()
    for i in t:
        done = truncated = False
        obs, info = env.reset()
        # set the envviewr in the env
        if args.render:
            env.render()
        # env.viewer.set_agent_display(display_action)
        # env.viewer.set_agent_display(functools.partial(display_vehicles, env=env))
        skip_run = False

        if args.initial_pos != "":
            load_veh = np.load(args.initial_pos, allow_pickle=True)

            env.road.vehicles = load_veh
            # for v in env.road.vehicles:
            #     v.ACC_MAX = 0.5
            #     v.DEACC_MAX = -0.5
            # v.LANE_CHANGE_MAX_BRAKING_IMPOSED = 10.0
            #     v.MAX_STEERING_ANGLE = np.pi / 3

            # env.road.vehicles = [env.road.vehicles[0]]
            env.set_vehicle(env.road.vehicles[0])

        # position_list = []
        # action_buffer = []
        while not (done or truncated):
            if not args.mobil:
                start = time.time()
                action, _states = model.predict(obs, deterministic=True)
                print(f"Inference took {time.time() - start}")

                # t_obs = torch.tensor(obs)
                # t_obs = t_obs[None, :]
                # action_prob, action_log_prob = model.policy.actor.action_log_prob(t_obs)
                # global last_action_prob
                # last_action_prob = action_prob.detach().numpy()
                # action_buffer.append(last_action_prob[0])
            else:
                action = None

            # action_map = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
            # print(action_map[action])
            start = time.time()
            # profiler.enable()
            obs, reward, done, truncated, info = env.step(action)
            # profiler.disable()
            print(f"Simulation took {time.time() - start}")

            global last_observation
            last_observation = obs
            global last_info
            last_info = info

            speed += info["average_speed"]
            road_speed += info["average_road_speed"]
            total_steps += 1

            if args.traj_dir != "":
                veh_pos = info["vehicle_position"][0]
                position_list.append(veh_pos.copy())

            if args.render:
                start = time.time()
                env.render()
                print(f"Render took {time.time() - start}")
                # time.sleep(0.1)

            # also end the episode when another vehicle crashed
            if info["other_crashes"] and not info["crashed"]:
                skip_run = True

        if skip_run:
            continue

        if info["other_crashes"] and not info["crashed"]:
            other_crashes += 1

        if info["crashed"]:
            crashes += 1
            # only save trajectories if we didn't load any in the first place
            # if args.initial_pos == "":
            #     np.save(f"initial_pos_{i}.npy", env.road.initial_vehicles)

        if info["merged"]:  # and not info["other_crashes"]:
            sucessfull_merges += 1

        j += 1

        # if args.traj_dir != '':
        # np.save(f"{args.traj_dir}/pos_{i}.npy", np.array(position_list))

        # print(f"Episode done crashed:{info['crashed']}")
        # print(f"Current crashrate {crashes/(i+1)}")
        # t.set_description(f"Crashrate {crashes/(i+1)} Other crashes {other_crashes/(i+1)}")
        # t.set_description(f"Crashrate {crashes/(i+1)} Mergerate {sucessfull_merges/(i+1-other_crashes)}")
        t.set_description(
            f"Crashrate {crashes/(j+1)} Mergerate {sucessfull_merges/(j+1)} other_crashes {other_crashes/(j+1)}"
        )

    # profiler.disable()
    # stats = pstats.Stats(profiler)
    # stats.dump_stats("profile_commonroad.log")
    # stats = pstats.Stats(profiler)
    # stats.dump_stats("train_prof_laptop_python308.log")
    end = time.time()
    print(f"Took {(end-start)/j}")
    print(f"Took {(end-start)/total_steps} per step")
    print(f"Crashrate {crashes/num_tries}")
    print(f"Average ego vehicle speed {speed/total_steps}")
    print(f"Average speed of all cars {road_speed/total_steps}")


if __name__ == "__main__":
    torch.set_num_threads(2)
    args = parse_args()
    eval_policy(args)
