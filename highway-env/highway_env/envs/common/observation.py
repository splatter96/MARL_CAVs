from typing import List, Dict, TYPE_CHECKING, Optional, Union
from gymnasium import spaces
import gymnasium as gym

gym.logger.set_level(40)
import numpy as np
import pandas as pd

from highway_env import utils
from highway_env.road.lane import AbstractLane
from highway_env.vehicle.controller import MDPVehicle

if TYPE_CHECKING:
    from highway_env.envs.common.abstract import AbstractEnv


class ObservationType(object):
    def __init__(self, env: "AbstractEnv", **kwargs) -> None:
        self.env = env
        self.__observer_vehicle = None

    def space(self) -> spaces.Space:
        """Get the observation space."""
        raise NotImplementedError()

    def observe(self):
        """Get an observation of the environment state."""
        raise NotImplementedError()

    @property
    def observer_vehicle(self):
        """
        The vehicle observing the scene.

        If not set, the first controlled vehicle is used by default.
        """
        return self.__observer_vehicle or self.env.vehicle

    @observer_vehicle.setter
    def observer_vehicle(self, vehicle):
        self.__observer_vehicle = vehicle


class GrayscaleObservation(ObservationType):
    """
    An observation class that collects directly what the simulator renders

    Also stacks the collected frames as in the nature DQN.
    Specific keys are expected in the configuration dictionary passed.

    Example of observation dictionary in the environment config:
        observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],  #weights for RGB conversion,
            "stack_size": 4,
            "observation_shape": (84, 84)
        }

    Also, the screen_height and screen_width of the environment should match the
    expected observation_shape.
    """

    def __init__(self, env: "AbstractEnv", config: dict) -> None:
        super().__init__(env)
        self.config = config
        self.observation_shape = config["observation_shape"]
        self.shape = self.observation_shape + (config["stack_size"],)
        self.state = np.zeros(self.shape)

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(shape=self.shape, low=0, high=1, dtype=np.float32)
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        new_obs = self._record_to_grayscale()
        new_obs = np.reshape(new_obs, self.observation_shape)
        self.state = np.roll(self.state, -1, axis=-1)
        self.state[:, :, -1] = new_obs
        return self.state

    def _record_to_grayscale(self) -> np.ndarray:
        # TODO: center rendering on the observer vehicle
        raw_rgb = self.env.render("rgb_array")
        return np.dot(raw_rgb[..., :3], self.config["weights"])


class TimeToCollisionObservation(ObservationType):
    def __init__(self, env: "AbstractEnv", horizon: int = 10, **kwargs: dict) -> None:
        super().__init__(env)
        self.horizon = horizon

    def space(self) -> spaces.Space:
        try:
            return spaces.Box(
                shape=self.observe().shape, low=0, high=1, dtype=np.float32
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(
                (3, 3, int(self.horizon * self.env.config["policy_frequency"]))
            )
        grid = compute_ttc_grid(
            self.env,
            vehicle=self.observer_vehicle,
            time_quantization=1 / self.env.config["policy_frequency"],
            horizon=self.horizon,
        )
        padding = np.ones(np.shape(grid))
        padded_grid = np.concatenate([padding, grid, padding], axis=1)
        obs_lanes = 3
        l0 = grid.shape[1] + self.observer_vehicle.lane_index[2] - obs_lanes // 2
        lf = grid.shape[1] + self.observer_vehicle.lane_index[2] + obs_lanes // 2
        clamped_grid = padded_grid[:, l0 : lf + 1, :]
        repeats = np.ones(clamped_grid.shape[0])
        repeats[np.array([0, -1])] += clamped_grid.shape[0]
        padded_grid = np.repeat(clamped_grid, repeats.astype(int), axis=0)
        obs_speeds = 3
        v0 = grid.shape[0] + self.observer_vehicle.speed_index - obs_speeds // 2
        vf = grid.shape[0] + self.observer_vehicle.speed_index + obs_speeds // 2
        clamped_grid = padded_grid[v0 : vf + 1, :, :]
        return clamped_grid


class KinematicObservation(ObservationType):
    """Observe the kinematics of nearby vehicles."""

    FEATURES: List[str] = ["presence", "x", "y", "vx", "vy"]

    def __init__(
        self,
        env: "AbstractEnv",
        features: List[str] = None,
        vehicles_count: int = 5,
        features_range: Dict[str, List[float]] = None,
        absolute: bool = False,
        order: str = "sorted",
        normalize: bool = True,
        clip: bool = False,
        see_behind: bool = True,
        observe_intentions: bool = False,
        weave_start_x: float = 230.0,
        weave_end_x: float = 310.0,
        **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        :param absolute: Use absolute coordinates for surrounding vehicles
        :param order: Order of observed vehicles. Values: sorted, shuffled
        :param normalize: Should the observation be normalized
        :param clip: Should the value be clipped in the desired range
        :param see_behind: Should the observation contain vehicles behind
        :param observe_intentions: Observe destinations of other vehicles
        :param weave_start_x: Longitudinal start of the weaving section
        :param weave_end_x: Longitudinal end of the weaving section
        """
        super().__init__(env)

        self.features = features or self.FEATURES
        self.vehicles_count = vehicles_count
        self.features_range = features_range
        self.absolute = absolute
        self.order = order
        self.normalize = normalize
        self.clip = clip
        self.see_behind = see_behind
        self.observe_intentions = observe_intentions

        self.weave_start_x = float(weave_start_x)
        self.weave_end_x = float(weave_end_x)

        if self.weave_end_x <= self.weave_start_x:
            raise ValueError(
                "weave_end_x must be greater than weave_start_x"
            )

    def space(self) -> spaces.Space:
        return spaces.Box(
            shape=(self.vehicles_count, len(self.features)),
            low=-1,
            high=1,
            dtype=np.float32,
        )

    def normalize_obs2(self, data: List[dict]):
        if not self.features_range:
            self.features_range = {
                "x": [-5.0 * MDPVehicle.SPEED_MAX, 5.0 * MDPVehicle.SPEED_MAX],
                "y": [-12, 12],
                "vx": [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
                "vy": [-1.5 * MDPVehicle.SPEED_MAX, 1.5 * MDPVehicle.SPEED_MAX],
            }

        for veh in data:
            for feature, _ in veh.items():
                if feature in self.features_range.keys():
                    veh[feature] = utils.lmap(
                        veh[feature],
                        [
                            self.features_range[feature][0],
                            self.features_range[feature][1],
                        ],
                        [-1, 1],
                    )

        return data

    def _weave_progress(self) -> float:
        """
        Calculate normalized progress through the weaving section.

        0 -> start of weaving section
        1 -> end of weaving section

        Values outside the weaving section are clipped.
        """
        x = float(self.observer_vehicle.position[0])

        progress = (
            (x - self.weave_start_x)
            / (self.weave_end_x - self.weave_start_x)
        )

        return float(np.clip(progress, 0.0, 1.0))

    def observe(self) -> np.ndarray:
        if not self.env.road:
            self.observed_vehicles = []
            self.observed_vehicle_ids = []
            return np.zeros(self.space().shape)

        # Collect nearby traffic. Keep the exact selected vehicle objects so
        # evaluation/rendering code can visualize precisely which vehicles
        # were visible to the policy for this observation.
        close_vehicles = self.env.road.close_vehicles_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=self.vehicles_count - 1,
            see_behind=self.see_behind,
        )

        selected_close_vehicles = list(
            close_vehicles[-self.vehicles_count + 1 :]
        ) if close_vehicles else []

        self.observed_vehicles = [self.observer_vehicle] + selected_close_vehicles
        self.observed_vehicle_ids = [
            getattr(vehicle, "id", None)
            for vehicle in self.observed_vehicles
        ]

        obs_list = []


        # Add ego-vehicle
        obs = self.observer_vehicle.to_dict()
        # extract only the features we want
        obs = {k: obs[k] for k in self.features if k in obs}
        obs_list.append(obs)

        if selected_close_vehicles:
            origin = self.observer_vehicle if not self.absolute else None

            close_veh = [
                v.to_dict(origin, observe_intentions=self.observe_intentions)
                for v in selected_close_vehicles
            ]
            # extract only the features we want
            for idx, veh in enumerate(close_veh):
                close_veh[idx] = {k: veh[k] for k in self.features if k in veh}
                obs_list.append(close_veh[idx])

        # Normalize and clip
        if self.normalize:
            obs_list = self.normalize_obs2(obs_list)

        # --------------------------------------------------------------
        # Replace ONLY the ego x position with weaving progress.
        #
        # This is intentionally done AFTER normalization so that:
        #
        # - surrounding vehicle relative x values are unchanged
        # - ego y, vx and vy are unchanged
        # - all normalization behavior remains unchanged
        # - p_weave is directly represented in [0, 1]
        # --------------------------------------------------------------

        if "x" in self.features:
            obs_list[0]["x"] = self._weave_progress()

        #print(f"obs_list: {obs_list}")


        # Fill missing rows
        if len(obs_list) < self.vehicles_count:
            empty_row = {k: 0 for k in self.features}
            for i in range(self.vehicles_count - len(obs_list)):
                obs_list.append(empty_row)

        # Convert to 2D Array
        res = [[item.get(key, "") for key in self.features] for item in obs_list]

        return res

# class KinematicObservation(ObservationType):
#     """
#     Semantic weaving observation.

#     Observation shape: (7, 5)

#     Row 0 (ego):
#         [p_weave, Lleft, Lright, vx, vy]

#     Rows 1-6 (surrounding vehicles):
#         [presence, relative_x, relative_y, relative_vx, relative_vy]

#     Surrounding-vehicle row order:
#         1: front vehicle in current lane
#         2: rear vehicle in current lane
#         3: front vehicle in left lane
#         4: rear vehicle in left lane
#         5: front vehicle in right lane
#         6: rear vehicle in right lane

#     Lane convention follows highway-env:
#         lane_id - 1 -> left lane
#         lane_id + 1 -> right lane
#     """

#     FEATURES: List[str] = ["presence", "x", "y", "vx", "vy"]
#     VEHICLES_COUNT = 7

#     def __init__(
#         self,
#         env: "AbstractEnv",
#         features: List[str] = None,
#         vehicles_count: int = 7,
#         features_range: Dict[str, List[float]] = None,
#         absolute: bool = False,
#         order: str = "sorted",
#         normalize: bool = True,
#         clip: bool = False,
#         see_behind: bool = True,
#         observe_intentions: bool = False,
#         weave_start_x: float = 230.0,
#         weave_end_x: float = 310.0,
#         **kwargs: dict,
#     ) -> None:
#         super().__init__(env)

#         # The observation layout is fixed to:
#         # 1 ego vehicle + 6 semantically selected surrounding vehicles.
#         #
#         # The arguments features, vehicles_count, absolute, order and
#         # see_behind are retained for compatibility with existing
#         # highway-env configuration dictionaries.
#         self.features = self.FEATURES
#         self.vehicles_count = self.VEHICLES_COUNT

#         self.features_range = (
#             dict(features_range) if features_range is not None else None
#         )

#         self.absolute = absolute
#         self.order = order
#         self.normalize = normalize
#         self.clip = clip
#         self.see_behind = see_behind
#         self.observe_intentions = observe_intentions

#         self.weave_start_x = float(weave_start_x)
#         self.weave_end_x = float(weave_end_x)

#         if self.weave_end_x <= self.weave_start_x:
#             raise ValueError(
#                 "weave_end_x must be larger than weave_start_x"
#             )

#     def space(self) -> spaces.Space:
#         return spaces.Box(
#             shape=(self.VEHICLES_COUNT, len(self.FEATURES)),
#             low=-1,
#             high=1,
#             dtype=np.float32,
#         )

#     def _ensure_features_range(self) -> None:
#         if self.features_range is None:
#             self.features_range = {}

#         self.features_range.setdefault(
#             "x",
#             [
#                 -5.0 * MDPVehicle.SPEED_MAX,
#                 5.0 * MDPVehicle.SPEED_MAX,
#             ],
#         )

#         self.features_range.setdefault(
#             "y",
#             [-12.0, 12.0],
#         )

#         self.features_range.setdefault(
#             "vx",
#             [
#                 -1.5 * MDPVehicle.SPEED_MAX,
#                 1.5 * MDPVehicle.SPEED_MAX,
#             ],
#         )

#         self.features_range.setdefault(
#             "vy",
#             [
#                 -1.5 * MDPVehicle.SPEED_MAX,
#                 1.5 * MDPVehicle.SPEED_MAX,
#             ],
#         )

#     def _normalize_value(
#         self,
#         value: float,
#         feature: str,
#     ) -> float:
#         self._ensure_features_range()

#         value = utils.lmap(
#             value,
#             self.features_range[feature],
#             [-1, 1],
#         )

#         if self.clip:
#             value = np.clip(value, -1, 1)

#         return float(value)

#     def _weave_progress(self) -> float:
#         """
#         Return normalized ego progress through the weaving section.

#         p_weave = 0 at weave_start_x
#         p_weave = 1 at weave_end_x

#         Values before/after the weaving section are clipped to [0, 1].
#         """
#         x = float(self.observer_vehicle.position[0])

#         progress = (
#             (x - self.weave_start_x)
#             / (self.weave_end_x - self.weave_start_x)
#         )

#         return float(np.clip(progress, 0.0, 1.0))

#     def _adjacent_lane_indices(self):
#         """
#         Return the lane indices directly to the left and right of ego.

#         highway-env uses decreasing lane IDs for moving left and
#         increasing lane IDs for moving right.
#         """
#         current = self.observer_vehicle.lane_index

#         if current is None:
#             return None, None

#         side_lanes = set(
#             self.env.road.network.all_side_lanes(current)
#         )

#         _from, _to, lane_id = current

#         left_lane = (_from, _to, lane_id - 1)
#         right_lane = (_from, _to, lane_id + 1)

#         if left_lane not in side_lanes:
#             left_lane = None

#         if right_lane not in side_lanes:
#             right_lane = None

#         return left_lane, right_lane

#     def _front_rear_in_lane(self, lane_index):
#         """
#         Find the closest front and rear vehicles in a specified lane.

#         The search is performed independently for every lane slot so that
#         a relevant front/rear vehicle cannot be omitted just because six
#         other vehicles happen to be geometrically closer.

#         Only vehicles within PERCEPTION_DISTANCE are considered.
#         """
#         if lane_index is None:
#             return None, None

#         ego = self.observer_vehicle

#         front_vehicle, rear_vehicle = self.env.road.surrounding_vehicles(ego, lane_index=lane_index)
#         # front_vehicle, rear_vehicle = self.env.road.neighbour_vehicles(ego, lane_index=lane_index)

#         # front_vehicle = None
#         # rear_vehicle = None

#         # closest_front_x = np.inf
#         # closest_rear_x = -np.inf

#         # for vehicle in self.env.road.vehicles:
#         #     if vehicle is ego:
#         #         continue

#         #     if vehicle.lane_index != lane_index:
#         #         continue

#         #     distance = np.linalg.norm(
#         #         vehicle.position - ego.position
#         #     )

#         #     if distance > self.env.PERCEPTION_DISTANCE:
#         #         continue

#         #     relative = vehicle.to_dict(
#         #         ego,
#         #         observe_intentions=self.observe_intentions,
#         #     )

#         #     dx = float(relative["x"])

#         #     # Vehicle in front
#         #     if dx >= 0.0:
#         #         if dx < closest_front_x:
#         #             closest_front_x = dx
#         #             front_vehicle = vehicle

#         #     # Vehicle behind
#         #     else:
#         #         if dx > closest_rear_x:
#         #             closest_rear_x = dx
#         #             rear_vehicle = vehicle

#         return front_vehicle, rear_vehicle

#     def _vehicle_row(self, vehicle) -> np.ndarray:
#         """
#         Create one surrounding-vehicle observation:

#         [
#             presence,
#             relative_x,
#             relative_y,
#             relative_vx,
#             relative_vy,
#         ]
#         """
#         if vehicle is None:
#             return np.zeros(
#                 len(self.FEATURES),
#                 dtype=np.float32,
#             )

#         data = vehicle.to_dict(
#             self.observer_vehicle,
#             observe_intentions=self.observe_intentions,
#         )

#         row = np.array(
#             [
#                 1.0,
#                 float(data["x"]),
#                 float(data["y"]),
#                 float(data["vx"]),
#                 float(data["vy"]),
#             ],
#             dtype=np.float32,
#         )

#         if self.normalize:
#             row[1] = self._normalize_value(
#                 row[1],
#                 "x",
#             )
#             row[2] = self._normalize_value(
#                 row[2],
#                 "y",
#             )
#             row[3] = self._normalize_value(
#                 row[3],
#                 "vx",
#             )
#             row[4] = self._normalize_value(
#                 row[4],
#                 "vy",
#             )

#         return row

#     def observe(self) -> np.ndarray:
#         if not self.env.road:
#             return np.zeros(
#                 self.space().shape,
#                 dtype=np.float32,
#             )

#         ego = self.observer_vehicle

#         current_lane = ego.lane_index
#         left_lane, right_lane = (
#             self._adjacent_lane_indices()
#         )

#         # --------------------------------------------------------------
#         # Ego observation
#         # --------------------------------------------------------------

#         ego_dict = ego.to_dict()

#         ego_vx = float(ego_dict["vx"])
#         ego_vy = float(ego_dict["vy"])

#         if self.normalize:
#             ego_vx = self._normalize_value(
#                 ego_vx,
#                 "vx",
#             )
#             ego_vy = self._normalize_value(
#                 ego_vy,
#                 "vy",
#             )

#         ego_row = np.array(
#             [
#                 self._weave_progress(),
#                 1.0 if left_lane is not None else 0.0,
#                 1.0 if right_lane is not None else 0.0,
#                 ego_vx,
#                 ego_vy,
#             ],
#             dtype=np.float32,
#         )

#         # --------------------------------------------------------------
#         # Current lane
#         # --------------------------------------------------------------

#         front_current, rear_current = (
#             self._front_rear_in_lane(
#                 current_lane
#             )
#         )

#         # --------------------------------------------------------------
#         # Left lane
#         # --------------------------------------------------------------

#         front_left, rear_left = (
#             self._front_rear_in_lane(
#                 left_lane
#             )
#         )

#         # --------------------------------------------------------------
#         # Right lane
#         # --------------------------------------------------------------

#         front_right, rear_right = (
#             self._front_rear_in_lane(
#                 right_lane
#             )
#         )

#         # --------------------------------------------------------------
#         # Assemble observation
#         #
#         # Row 0: ego
#         # Row 1: front current
#         # Row 2: rear current
#         # Row 3: front left
#         # Row 4: rear left
#         # Row 5: front right
#         # Row 6: rear right
#         # --------------------------------------------------------------

#         observation = np.vstack(
#             [
#                 ego_row,
#                 self._vehicle_row(front_current),
#                 self._vehicle_row(rear_current),
#                 self._vehicle_row(front_left),
#                 self._vehicle_row(rear_left),
#                 self._vehicle_row(front_right),
#                 self._vehicle_row(rear_right),
#             ]
#         )

#         return observation.astype(np.float32)


class OccupancyGridObservation(ObservationType):
    """Observe an occupancy grid of nearby vehicles."""

    FEATURES: List[str] = ["presence", "vx", "vy"]
    GRID_SIZE: List[List[float]] = [[-5.5 * 5, 5.5 * 5], [-5.5 * 5, 5.5 * 5]]
    GRID_STEP: List[int] = [5, 5]

    def __init__(
        self,
        env: "AbstractEnv",
        features: Optional[List[str]] = None,
        grid_size: Optional[List[List[float]]] = None,
        grid_step: Optional[List[int]] = None,
        features_range: Dict[str, List[float]] = None,
        absolute: bool = False,
        **kwargs: dict,
    ) -> None:
        """
        :param env: The environment to observe
        :param features: Names of features used in the observation
        :param vehicles_count: Number of observed vehicles
        """
        super().__init__(env)
        self.features = features if features is not None else self.FEATURES
        self.grid_size = (
            np.array(grid_size) if grid_size is not None else np.array(self.GRID_SIZE)
        )
        self.grid_step = (
            np.array(grid_step) if grid_step is not None else np.array(self.GRID_STEP)
        )
        grid_shape = np.asarray(
            np.floor((self.grid_size[:, 1] - self.grid_size[:, 0]) / grid_step),
            dtype=np.int,
        )
        self.grid = np.zeros((len(self.features), *grid_shape))
        self.features_range = features_range
        self.absolute = absolute

    def space(self) -> spaces.Space:
        return spaces.Box(shape=self.grid.shape, low=-1, high=1, dtype=np.float32)

    def normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize the observation values.

        For now, assume that the road is straight along the x axis.
        :param Dataframe df: observation data
        """
        if not self.features_range:
            self.features_range = {
                "vx": [-2 * MDPVehicle.SPEED_MAX, 2 * MDPVehicle.SPEED_MAX],
                "vy": [-2 * MDPVehicle.SPEED_MAX, 2 * MDPVehicle.SPEED_MAX],
            }
        for feature, f_range in self.features_range.items():
            if feature in df:
                df[feature] = utils.lmap(df[feature], [f_range[0], f_range[1]], [-1, 1])
        return df

    def observe(self) -> np.ndarray:
        if not self.env.road:
            return np.zeros(self.space().shape)

        if self.absolute:
            raise NotImplementedError()
        else:
            # Add nearby traffic
            self.grid.fill(0)
            df = pd.DataFrame.from_records(
                [v.to_dict(self.observer_vehicle) for v in self.env.road.vehicles]
            )
            # Normalize
            df = self.normalize(df)
            # Fill-in features
            for layer, feature in enumerate(self.features):
                for _, vehicle in df.iterrows():
                    x, y = vehicle["x"], vehicle["y"]
                    # Recover unnormalized coordinates for cell index
                    if "x" in self.features_range:
                        x = utils.lmap(
                            x,
                            [-1, 1],
                            [self.features_range["x"][0], self.features_range["x"][1]],
                        )
                    if "y" in self.features_range:
                        y = utils.lmap(
                            y,
                            [-1, 1],
                            [self.features_range["y"][0], self.features_range["y"][1]],
                        )
                    cell = (
                        int((x - self.grid_size[0, 0]) / self.grid_step[0]),
                        int((y - self.grid_size[1, 0]) / self.grid_step[1]),
                    )
                    if (
                        0 <= cell[1] < self.grid.shape[-2]
                        and 0 <= cell[0] < self.grid.shape[-1]
                    ):
                        self.grid[layer, cell[1], cell[0]] = vehicle[feature]
            # Clip
            obs = np.clip(self.grid, -1, 1)
            return obs


class KinematicsGoalObservation(KinematicObservation):
    def __init__(self, env: "AbstractEnv", scales: List[float], **kwargs: dict) -> None:
        self.scales = np.array(scales)
        super().__init__(env, **kwargs)

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(
                dict(
                    desired_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["desired_goal"].shape,
                        dtype=np.float32,
                    ),
                    achieved_goal=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["achieved_goal"].shape,
                        dtype=np.float32,
                    ),
                    observation=spaces.Box(
                        -np.inf,
                        np.inf,
                        shape=obs["observation"].shape,
                        dtype=np.float32,
                    ),
                )
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        if not self.observer_vehicle:
            return {
                "observation": np.zeros((len(self.features),)),
                "achieved_goal": np.zeros((len(self.features),)),
                "desired_goal": np.zeros((len(self.features),)),
            }

        obs = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        )
        goal = np.ravel(
            pd.DataFrame.from_records([self.env.goal.to_dict()])[self.features]
        )
        obs = {
            "observation": obs / self.scales,
            "achieved_goal": obs / self.scales,
            "desired_goal": goal / self.scales,
        }
        return obs


class AttributesObservation(ObservationType):
    def __init__(
        self, env: "AbstractEnv", attributes: List[str], **kwargs: dict
    ) -> None:
        self.env = env
        self.attributes = attributes

    def space(self) -> spaces.Space:
        try:
            obs = self.observe()
            return spaces.Dict(
                {
                    attribute: spaces.Box(
                        -np.inf, np.inf, shape=obs[attribute].shape, dtype=np.float32
                    )
                    for attribute in self.attributes
                }
            )
        except AttributeError:
            return spaces.Space()

    def observe(self) -> Dict[str, np.ndarray]:
        return {
            attribute: getattr(self.env, attribute) for attribute in self.attributes
        }


class MultiAgentObservation(ObservationType):
    def __init__(self, env: "AbstractEnv", observation_config: dict, **kwargs) -> None:
        super().__init__(env)
        self.observation_config = observation_config
        self.agents_observation_types = []
        for vehicle in self.env.controlled_vehicles:
            obs_type = observation_factory(self.env, self.observation_config)
            obs_type.observer_vehicle = vehicle
            self.agents_observation_types.append(obs_type)

    def space(self) -> spaces.Space:
        return spaces.Tuple(
            [obs_type.space() for obs_type in self.agents_observation_types]
        )

    def observe(self) -> tuple:
        return tuple(obs_type.observe() for obs_type in self.agents_observation_types)


def observation_factory(env: "AbstractEnv", config: dict) -> ObservationType:
    if config["type"] == "TimeToCollision":
        return TimeToCollisionObservation(env, **config)
    elif config["type"] == "Kinematics":
        return KinematicObservation(env, **config)
    elif config["type"] == "OccupancyGrid":
        return OccupancyGridObservation(env, **config)
    elif config["type"] == "KinematicsGoal":
        return KinematicsGoalObservation(env, **config)
    elif config["type"] == "GrayscaleObservation":
        return GrayscaleObservation(env, config)
    elif config["type"] == "AttributesObservation":
        return AttributesObservation(env, **config)
    elif config["type"] == "MultiAgentObservation":
        return MultiAgentObservation(env, **config)
    else:
        raise ValueError("Unknown observation type")
