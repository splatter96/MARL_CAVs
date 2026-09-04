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


def _get_observed_vehicles():
    """Return the exact vehicles used in the latest KinematicObservation.

    KinematicObservation stores these when ``observe()`` is called. Row 0 is
    the ego vehicle and rows 1..N are the surrounding vehicles supplied to the
    policy. Missing/padded rows are not represented by a vehicle object.
    """
    global render_env

    if render_env is None:
        return []

    observation_type = getattr(render_env, "observation_type", None)
    if observation_type is None:
        return []

    return list(getattr(observation_type, "observed_vehicles", []))


def display_observed_vehicles(action_surface, sim_surface):
    """Visualize the policy observation and the ego vehicle's target lane.

    The labels correspond to rows of the KinematicObservation:

        OBS 0 = ego row
        OBS 1..N = surrounding-vehicle rows

    Only real vehicle rows are drawn; zero-padded rows have no corresponding
    vehicle and therefore no circle in the road view.

    In addition, the current ``target_lane_index`` of the ego vehicle is
    highlighted directly in the road view. This makes it easier to relate the
    policy's selected action to the lane toward which the controlled vehicle
    is currently steering.
    """
    global render_env

    if render_env is None:
        return

    observed_vehicles = _get_observed_vehicles()
    if not observed_vehicles:
        return

    ego = observed_vehicles[0]

    # Use one color for ego and one for every surrounding vehicle selected by
    # KinematicObservation. Keeping all observed traffic the same color makes
    # it easy to distinguish "visible to policy" from normal renderer colors.
    ego_color = (255, 80, 180)
    observed_color = (0, 220, 120)
    text_color = (20, 20, 20)

    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 19)

    # ------------------------------------------------------------------
    # Observation summary panel
    # ------------------------------------------------------------------
    observation_type = getattr(render_env, "observation_type", None)
    vehicles_count = getattr(observation_type, "vehicles_count", len(observed_vehicles))
    surrounding_count = max(0, len(observed_vehicles) - 1)
    padded_count = max(0, int(vehicles_count) - len(observed_vehicles))

    panel_x = 10
    panel_y = 10
    panel_width = 520
    # One additional line is reserved for the current target-lane index.
    panel_height = 78 + 20 * min(len(observed_vehicles), 8)

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

    title = font.render(
        f"Kinematic observation: {surrounding_count} surrounding vehicles",
        True,
        text_color,
    )
    action_surface.blit(title, (panel_x + 8, panel_y + 5))

    summary = small_font.render(
        f"Rows used: {len(observed_vehicles)}/{vehicles_count}  "
        f"zero-padded: {padded_count}",
        True,
        text_color,
    )
    action_surface.blit(summary, (panel_x + 8, panel_y + 28))

    # ------------------------------------------------------------------
    # Highlight the ego vehicle's current target lane
    # ------------------------------------------------------------------
    target_lane_index = getattr(ego, "target_lane_index", None)
    target_lane_color = (255, 215, 0)

    target_lane_text = "None" if target_lane_index is None else str(target_lane_index)
    target_line = small_font.render(
        f"Target lane: {target_lane_text}",
        True,
        target_lane_color,
    )
    action_surface.blit(target_line, (panel_x + 8, panel_y + 48))

    if target_lane_index is not None:
        try:
            target_lane = render_env.road.network.get_lane(target_lane_index)

            # Sample the complete lane centerline. Using lane.position() rather
            # than assuming a straight lane also works for curved/sine lanes.
            sample_count = max(2, int(np.ceil(float(target_lane.length) / 2.0)) + 1)
            longitudinal_samples = np.linspace(
                0.0,
                float(target_lane.length),
                sample_count,
            )

            target_pixels = []
            for longitudinal in longitudinal_samples:
                point = target_lane.position(float(longitudinal), 0.0)
                pixel = sim_surface.pos2pix(float(point[0]), float(point[1]))
                target_pixels.append((int(pixel[0]), int(pixel[1])))

            if len(target_pixels) >= 2:
                # Draw a broad dark outline first so the target lane remains
                # visible on both light and dark road markings.
                pygame.draw.lines(
                    sim_surface,
                    (40, 40, 40),
                    False,
                    target_pixels,
                    8,
                )
                pygame.draw.lines(
                    sim_surface,
                    target_lane_color,
                    False,
                    target_pixels,
                    5,
                )

            # Place the TARGET label close to the longitudinal position of ego
            # projected onto the target lane, rather than at a distant midpoint.
            ego_longitudinal, _ = target_lane.local_coordinates(ego.position)
            ego_longitudinal = float(
                np.clip(ego_longitudinal, 0.0, float(target_lane.length))
            )
            label_position = target_lane.position(ego_longitudinal, 0.0)
            label_pixel = sim_surface.pos2pix(
                float(label_position[0]),
                float(label_position[1]),
            )

            # target_label = font.render("TARGET", True, target_lane_color)
            # sim_surface.blit(
            #     target_label,
            #     (int(label_pixel[0]) + 10, int(label_pixel[1]) + 14),
            # )
        except Exception:
            # Visualization should never interfere with policy evaluation if a
            # temporary target-lane index cannot be resolved by the road graph.
            pass

    # ------------------------------------------------------------------
    # Draw every real observation row in the road view
    # ------------------------------------------------------------------
    ego_pixel = sim_surface.pos2pix(
        float(ego.position[0]),
        float(ego.position[1]),
    )

    for row_idx, vehicle in enumerate(observed_vehicles):
        vehicle_pixel = sim_surface.pos2pix(
            float(vehicle.position[0]),
            float(vehicle.position[1]),
        )

        color = ego_color if row_idx == 0 else observed_color
        label_text = "" if row_idx == 0 else f"{row_idx}"

        # Circle around the vehicle used in this observation row.
        pygame.draw.circle(
            sim_surface,
            color,
            (int(vehicle_pixel[0]), int(vehicle_pixel[1])),
            20,
            4,
        )

        # Connect surrounding observed vehicles to ego for quick inspection.
        if row_idx > 0:
            pygame.draw.line(
                sim_surface,
                color,
                (int(ego_pixel[0]), int(ego_pixel[1])),
                (int(vehicle_pixel[0]), int(vehicle_pixel[1])),
                2,
            )

        label = font.render(label_text, True, color)
        sim_surface.blit(
            label,
            (int(vehicle_pixel[0]) + 20, int(vehicle_pixel[1]) - 24),
        )

        vehicle_id = getattr(vehicle, "id", None)
        id_text = "None" if vehicle_id is None else str(vehicle_id)
        panel_line = small_font.render(
            f"OBS {row_idx}: vehicle id {id_text}",
            True,
            color,
        )
        action_surface.blit(
            panel_line,
            (panel_x + 10, panel_y + 70 + row_idx * 20),
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
        model = SACD.load(args.model)
        #model = DQN.load(args.model)
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
    ttc_current_front_values = []
    ttc_target_front_values = []
    ttc_target_rear_values = []

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
            viewer.set_agent_display(display_observed_vehicles)
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
            ttc_current_front_values.append(float(info["ttc_current_front"]))
            ttc_target_front_values.append(float(info["ttc_target_front"]))
            ttc_target_rear_values.append(float(info["ttc_target_rear"]))

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


    np.save("crash_positions.npy", np.array(crash_positions))
    np.save("actions.npy", action_buffer)

if __name__ == "__main__":
    torch.set_num_threads(2)
    args = parse_args()

    eval_policy(args)
