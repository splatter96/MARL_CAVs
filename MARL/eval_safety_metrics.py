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
from stable_baselines3 import DQN, PPO

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
        "--seed",
        type=int,
        required=False,
        default=42,
        help="Random seed for the model",
    )
    parser.add_argument(
        "--traj-dir",
        type=str,
        required=False,
        default="",
        help="directory where to save the trajectories",
    )
    parser.add_argument(
        "--metrics-dir",
        type=str,
        required=False,
        default=".",
        help=(
            "directory where the TTC metric numpy files are saved "
            "(default: current directory)"
        ),
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
render_env = None

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

def _format_ttc(value):
    """Format a TTC value for the renderer."""
    if value is None:
        return "n/a"

    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if np.isnan(value):
        return "n/a"
    if np.isinf(value):
        return "inf"
    return f"{value:.2f} s"


def _find_vehicle_by_id(vehicle_id):
    """Find a vehicle in the current rendered environment by its ID."""
    global render_env

    if render_env is None or vehicle_id is None:
        return None

    for vehicle in render_env.road.vehicles:
        current_id = getattr(vehicle, "id", None)

        # Exact comparison first. The string fallback also handles cases in
        # which NumPy/scalar types are used in the info dictionary.
        if current_id == vehicle_id or str(current_id) == str(vehicle_id):
            return vehicle

    return None


def display_ttc(action_surface, sim_surface):
    """Visualize TTC values and the vehicles used to compute them.

    TTC information is shown in ``action_surface``. The corresponding vehicles
    are highlighted directly in ``sim_surface`` and connected to the ego vehicle
    with a line.

    Labels:
        CF = current-lane front vehicle
        TF = target-lane front vehicle
        TR = target-lane rear vehicle
    """
    global last_info, render_env

    if last_info is None or render_env is None:
        return

    ego = getattr(render_env, "vehicle", None)
    if ego is None:
        return

    # Distinct colors for the three TTC relations.
    metrics = [
        {
            "label": "CF",
            "name": "Current front",
            "ttc_key": "ttc_current_front",
            "id_key": "current_front_id",
            "color": (255, 190, 0),
        },
        {
            "label": "TF",
            "name": "Target front",
            "ttc_key": "ttc_target_front",
            "id_key": "target_front_id",
            "color": (0, 210, 255),
        },
        {
            "label": "TR",
            "name": "Target rear",
            "ttc_key": "ttc_target_rear",
            "id_key": "target_rear_id",
            "color": (255, 70, 220),
        },
    ]

    # ------------------------------------------------------------------
    # Text panel with the TTC values
    # ------------------------------------------------------------------
    font = pygame.font.Font(None, 26)
    small_font = pygame.font.Font(None, 20)

    panel_x = 10
    panel_y = 10
    panel_width = 390
    panel_height = 100

    pygame.draw.rect(
        action_surface,
        (245, 245, 245),
        (panel_x, panel_y, panel_width, panel_height),
    )
    pygame.draw.rect(
        action_surface,
        (40, 40, 40),
        (panel_x, panel_y, panel_width, panel_height),
        1,
    )

    title = font.render("Time-to-Collision", True, (20, 20, 20))
    action_surface.blit(title, (panel_x + 8, panel_y + 5))

    for i, metric in enumerate(metrics):
        ttc = last_info.get(metric["ttc_key"], None)
        vehicle_id = last_info.get(metric["id_key"], None)

        id_text = "None" if vehicle_id is None else str(vehicle_id)
        text = (
            f'{metric["label"]}: {_format_ttc(ttc)}   '
            f'(vehicle {id_text})'
        )

        rendered = small_font.render(text, True, metric["color"])
        action_surface.blit(
            rendered,
            (panel_x + 10, panel_y + 35 + i * 20),
        )

    # ------------------------------------------------------------------
    # Highlight the vehicles used for the TTC calculation in the road view
    # ------------------------------------------------------------------
    ego_pixel = sim_surface.pos2pix(
        float(ego.position[0]),
        float(ego.position[1]),
    )

    for metric in metrics:
        vehicle_id = last_info.get(metric["id_key"], None)
        vehicle = _find_vehicle_by_id(vehicle_id)

        if vehicle is None:
            continue

        vehicle_pixel = sim_surface.pos2pix(
            float(vehicle.position[0]),
            float(vehicle.position[1]),
        )

        color = metric["color"]

        # Line from ego to the vehicle involved in this TTC measurement.
        pygame.draw.line(
            sim_surface,
            color,
            (int(ego_pixel[0]), int(ego_pixel[1])),
            (int(vehicle_pixel[0]), int(vehicle_pixel[1])),
            3,
        )

        # Marker around the TTC vehicle.
        pygame.draw.circle(
            sim_surface,
            color,
            (int(vehicle_pixel[0]), int(vehicle_pixel[1])),
            18,
            4,
        )

        # Put CF/TF/TR next to the highlighted vehicle.
        label = font.render(metric["label"], True, color)
        sim_surface.blit(
            label,
            (int(vehicle_pixel[0]) + 18, int(vehicle_pixel[1]) - 22),
        )


def eval_policy(args):
    global last_info, last_observation, render_env

    #env = gym.make("merge-single-agent-v0")

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
    render_env = env.unwrapped


    if not args.mobil:
        #model = SACD.load(args.model)
        model = DQN.load(args.model)
        # model.set_random_seed(21)
        model.set_random_seed(args.seed)

    # Number of successful crashes we aim for
    target_crashes = args.num_runs

    crashes = 0
    other_crashes = 0
    speed = 0
    road_speed = 0
    total_steps = 0
    sucessfull_merges = 0

    crash_positions = []

    # create the output directory if we need to save the trajectories
    if args.traj_dir != "" and not os.path.exists(args.traj_dir):
        os.mkdir(args.traj_dir)

    j = 0  # episode counter
    # Use tqdm without a predefined total; we update it each episode.
    t = tqdm(desc="Evaluation episodes", total=target_crashes)
    start = time.time()

    action_map = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
    action_buffer = []
    # Step-wise TTC values over the complete evaluation. These are kept for
    # distribution plots and threshold-based statistics.
    ttc_current_front_values = []
    ttc_target_front_values = []
    ttc_target_rear_values = []

    # Episode-wise minimum TTC values. One value is stored per evaluation
    # episode and interaction type. np.inf is preserved if no closing conflict
    # occurred for that interaction type during the episode.
    episode_min_ttc_current_front = []
    episode_min_ttc_target_front = []
    episode_min_ttc_target_rear = []

    # Accepted merge gaps measured by the environment at merge initiation and
    # reported once the merge is successfully completed. We store one value per
    # episode; unsuccessful episodes remain np.nan. This keeps the values aligned
    # with the evaluation episodes and allows np.nanmedian()/percentiles later.
    accepted_merge_gap_front = []
    accepted_merge_gap_rear = []
    successful_merge_flags = []

    os.makedirs(args.metrics_dir, exist_ok=True)


    # Run episodes until we have observed ``target_crashes`` crashes.
    # while crashes < target_crashes:
    while j < target_crashes:
        done = truncated = False
        obs, info = env.reset()
        last_observation = obs
        last_info = info

        if args.render:
            env.render()
            # env.viewer.set_agent_display(display_action)
            viewer = getattr(env, "viewer", None)
            if viewer is None:
                viewer = env.unwrapped.viewer
            viewer.set_agent_display(display_ttc)
        skip_run = False

        if args.initial_pos != "":
            load_veh = np.load(args.initial_pos, allow_pickle=True)

            env.road.vehicles = load_veh
            env.set_vehicle(env.road.vehicles[0])

        # ret = 0
        # position_list = []
        # action_buffer = []
        # initialise per-episode storage for trajectory positions if needed
        position_list = []

        # Per-episode TTC traces are used to compute the minimum TTC for each of
        # the three relevant interaction partners.
        episode_ttc_current_front = []
        episode_ttc_target_front = []
        episode_ttc_target_rear = []

        # The environment reports the accepted gap only after a successful merge,
        # but the stored values correspond to the instant at which the merge was
        # initiated. Keep np.nan if no successful merge occurs in this episode.
        episode_merge_gap_front = np.nan
        episode_merge_gap_rear = np.nan
        episode_merged = False

        while not (done or truncated):
            if not args.mobil:
                action, _states = model.predict(obs, deterministic=True)
                action_buffer.append(action)
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
            last_observation = obs
            last_info = info
            # ret += reward

            # global last_observation
            # last_observation = obs
            # global last_info
            # last_info = info

            speed += info["average_speed"]
            road_speed += info["average_road_speed"]
            total_steps += 1

            # Collect TTC metrics for every simulation step over the complete
            # evaluation. The environment reports np.inf when no closing
            # conflict exists, and those values are preserved in the files.
            ttc_current = float(info["ttc_current_front"])
            ttc_target_front = float(info["ttc_target_front"])
            ttc_target_rear = float(info["ttc_target_rear"])

            ttc_current_front_values.append(ttc_current)
            ttc_target_front_values.append(ttc_target_front)
            ttc_target_rear_values.append(ttc_target_rear)

            episode_ttc_current_front.append(ttc_current)
            episode_ttc_target_front.append(ttc_target_front)
            episode_ttc_target_rear.append(ttc_target_rear)

            # In the modified environment these fields become available on the
            # step on which a successful merge is first detected. The gap itself
            # was captured earlier, at the first LANE_LEFT action on the parallel
            # on-ramp section. Capture it here before later steps overwrite the
            # info values with np.nan again.
            if info.get("merged_now", False):
                episode_merge_gap_front = float(
                    info.get("accepted_merge_gap_front", np.nan)
                )
                episode_merge_gap_rear = float(
                    info.get("accepted_merge_gap_rear", np.nan)
                )
                episode_merged = True

            if args.traj_dir != "":
                veh_pos = info["vehicle_position"][0]
                position_list.append(veh_pos.copy())

            if args.render:
                env.render()
                time.sleep(0.1)

            # also end the episode when another vehicle crashed
            if info["other_crashes"] and not info["crashed"]:
                break
            #     skip_run = True

            #for v in env.unwrapped.road.vehicles:
                #if v.id == 6:
                    #print(f"Front: {v.speed}")
                #elif v.id==0:
                    #print(f"Ego: {v.speed}")

        # Store episode-level safety metrics before processing the terminal
        # outcome. Each minimum is taken over all simulation steps in the episode.
        # np.inf remains meaningful here: it denotes that no closing TTC conflict
        # occurred for that interaction type during the complete episode.
        episode_min_ttc_current_front.append(
            float(np.min(episode_ttc_current_front))
            if episode_ttc_current_front
            else np.inf
        )
        episode_min_ttc_target_front.append(
            float(np.min(episode_ttc_target_front))
            if episode_ttc_target_front
            else np.inf
        )
        episode_min_ttc_target_rear.append(
            float(np.min(episode_ttc_target_rear))
            if episode_ttc_target_rear
            else np.inf
        )

        accepted_merge_gap_front.append(float(episode_merge_gap_front))
        accepted_merge_gap_rear.append(float(episode_merge_gap_rear))
        successful_merge_flags.append(bool(episode_merged))

        # if skip_run:
        #     continue

        if info["other_crashes"] and not info["crashed"]:
            other_crashes += 1
            #if args.initial_pos == "":
                #np.save(f"initial_pos_{j}.npy", env.road.initial_vehicles)

        if info["crashed"]:
            # t.update(1)
            crashes += 1
            # save position of crash
            crash_positions.append(info["vehicle_position"][0])
            # only save trajectories if we didn't load any in the first place
            #if args.initial_pos == "":
                #np.save(f"initial_pos_{j}.npy", env.road.initial_vehicles)
            # np.save(f"action_before_crash_{j}.npy", action_buffer)
        # else:
        # np.save(f"action_without_crash_{j}.npy", action_buffer)

        if "merged" in info and info["merged"]:  # and not info["other_crashes"]:
            sucessfull_merges += 1

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


    metrics_dir = args.metrics_dir

    np.save(
        os.path.join(metrics_dir, "ttc_current_front.npy"),
        np.asarray(ttc_current_front_values, dtype=float),
    )
    np.save(
        os.path.join(metrics_dir, "ttc_target_front.npy"),
        np.asarray(ttc_target_front_values, dtype=float),
    )
    np.save(
        os.path.join(metrics_dir, "ttc_target_rear.npy"),
        np.asarray(ttc_target_rear_values, dtype=float),
    )

    # Episode-wise minimum TTC values used for the safety-margin table and the
    # episode-level TTC distribution figure.
    np.save(
        os.path.join(metrics_dir, "episode_min_ttc_current_front.npy"),
        np.asarray(episode_min_ttc_current_front, dtype=float),
    )
    np.save(
        os.path.join(metrics_dir, "episode_min_ttc_target_front.npy"),
        np.asarray(episode_min_ttc_target_front, dtype=float),
    )
    np.save(
        os.path.join(metrics_dir, "episode_min_ttc_target_rear.npy"),
        np.asarray(episode_min_ttc_target_rear, dtype=float),
    )

    # One entry per episode. Failed/non-completed merges are represented by
    # np.nan, so the accepted-gap statistics can later be computed directly with
    # np.nanmedian, np.nanpercentile, etc. The Boolean flag preserves which
    # episodes actually completed a merge.
    np.save(
        os.path.join(metrics_dir, "accepted_merge_gap_front.npy"),
        np.asarray(accepted_merge_gap_front, dtype=float),
    )
    np.save(
        os.path.join(metrics_dir, "accepted_merge_gap_rear.npy"),
        np.asarray(accepted_merge_gap_rear, dtype=float),
    )
    np.save(
        os.path.join(metrics_dir, "successful_merge.npy"),
        np.asarray(successful_merge_flags, dtype=bool),
    )

    # Convenience archive containing all safety-margin data in one file.
    np.savez(
        os.path.join(metrics_dir, "safety_margin_metrics.npz"),
        ttc_current_front=np.asarray(ttc_current_front_values, dtype=float),
        ttc_target_front=np.asarray(ttc_target_front_values, dtype=float),
        ttc_target_rear=np.asarray(ttc_target_rear_values, dtype=float),
        episode_min_ttc_current_front=np.asarray(
            episode_min_ttc_current_front, dtype=float
        ),
        episode_min_ttc_target_front=np.asarray(
            episode_min_ttc_target_front, dtype=float
        ),
        episode_min_ttc_target_rear=np.asarray(
            episode_min_ttc_target_rear, dtype=float
        ),
        accepted_merge_gap_front=np.asarray(accepted_merge_gap_front, dtype=float),
        accepted_merge_gap_rear=np.asarray(accepted_merge_gap_rear, dtype=float),
        successful_merge=np.asarray(successful_merge_flags, dtype=bool),
    )

    np.save("crash_positions.npy", np.array(crash_positions))
    np.save("actions.npy", action_buffer)

if __name__ == "__main__":
    torch.set_num_threads(2)
    args = parse_args()

    eval_policy(args)
