import numpy as np

from gymnasium.envs.registration import register

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv
from highway_env.road.lane import LineType, StraightLane, SineLane, HorizontalLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics

from highway_env.road.objects import Obstacle

from highway_env.vehicle.kinematics import Vehicle, RealVehicle


class SingleAgentMergeEnv(AbstractEnv):
    """
    A highway-env merge negotiation environment.

    The ego-vehicle is driving on a highway-env and approached a merge, with some vehicles incoming on the access ramp.
    It is rewarded for maintaining a high speed and avoiding collisions, but also making room for merging
    vehicles.
    """

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        cfg.update(
            {
                "duration": 15,  # time step
                "policy_frequency": 5,  # [Hz]
                #"policy_frequency": 15,  # [Hz]
                "reward_speed_range": [10, 30],
                "collision_reward": 200,
                "high_speed_reward": 1,
                "offramp_reward": 100,
                "HEADWAY_COST": 4,  # default=1
                "HEADWAY_TIME": 1.2,  # default=1.2[s]
                "MERGING_LANE_COST": 4,  # default=4
                "LANE_CHANGE_COST": 1,  # default=0.5
                "traffic_density": 1,  # easy or hard modes
                "use_weaving": True,
            }
        )
        return cfg

    def set_vehicle(self, veh):
        self.vehicle = veh

    def _reward(self, action: int) -> float:
        return self._agent_reward(action, self.vehicle)

    def step(self, action):
        """Step the environment and append weaving-safety metrics to ``info``.

        The additional metrics are evaluated in the state reached after the
        action has been simulated:

        * ``ttc_current_front``: TTC to the closest leader in the ego's
          current lane.
        * ``ttc_target_front``: TTC to the closest vehicle ahead in the
          target lane.
        * ``ttc_target_rear``: TTC to the closest vehicle approaching from
          behind in the target lane.
        * ``target_rear_induced_braking``: additional braking demand [m/s^2]
          imposed on the target-lane rear vehicle if the ego were inserted in
          front of it at the current state.

        TTC is ``np.inf`` if no relevant vehicle exists or the relative
        longitudinal velocity is not closing.  The induced-braking value is
        ``np.nan`` when no target-lane rear vehicle/model is available.
        """
        obs, reward, terminated, truncated, info = super().step(action)

        # Distinguish collisions in which another vehicle hits the ego from
        # behind. This is useful for open-loop trajectory replay (e.g. NGSIM),
        # where surrounding vehicles cannot react to the ego vehicle after it
        # changes lane.
        # info["rear_end_collision_by_other"] = self._rear_end_collision_by_other_vehicle()

        # if not getattr(self, "_merged_successfully", False):
        #     self._merged_successfully = self._is_successfully_merged()
        # info["merged"] = self._is_successfully_merged()

        #info.update(self._compute_safety_info())
        return obs, reward, terminated, truncated, info

    def _compute_safety_info(self):
        """Compute TTC and target-lane braking metrics for the ego vehicle."""
        ego = self.vehicle

        # ---- Current-lane leader -------------------------------------------------
        current_lane_index = ego.lane_index
        current_front = None
        current_front, _, current_front_gap, _ = self._lane_neighbours(
            ego, current_lane_index, include_next_front=True
        )
        ttc_current_front = self._front_ttc(
            ego,
            current_front,
            current_front_gap,
            current_lane_index,
        )

        # ---- Target-lane front/rear ---------------------------------------------
        target_front = None
        target_rear = None

        target_lane_index = self._get_target_lane_index()
        if target_lane_index is None:
            ttc_target_front = np.inf
            ttc_target_rear = np.inf
            induced_braking = np.nan
        else:
            target_front, target_rear, target_front_gap, target_rear_gap = (
                self._lane_neighbours(
                    ego, target_lane_index, include_next_front=True
                )
            )

            ttc_target_front = self._front_ttc(
                ego,
                target_front,
                target_front_gap,
                target_lane_index,
            )
            ttc_target_rear = self._rear_ttc(
                ego,
                target_rear,
                target_rear_gap,
                target_lane_index,
            )
            induced_braking = self._target_rear_induced_braking(
                ego,
                target_rear,
                target_front,
            )

        return {
            "ttc_current_front": float(ttc_current_front),
            "ttc_target_front": float(ttc_target_front),
            "ttc_target_rear": float(ttc_target_rear),
            "target_rear_induced_braking": float(induced_braking),
            "current_front_id": current_front.id if current_front is not None else None,
            "target_front_id": target_front.id if target_front is not None else None,
            "target_rear_id": target_rear.id if target_rear is not None else None,
        }

    def _get_target_lane_index(self):
        """Return the lane that is relevant as the ego's merge/lane-change target.

        In the weaving section the ego approaches on lane ("b", "c", 2) and
        merges left into ("b", "c", 1).  If the controlled vehicle already
        has a different explicit ``target_lane_index`` on the same road
        segment, that commanded lane takes precedence (e.g. a further lane
        change from lane 1 to lane 0).
        """
        current = self.vehicle.lane_index
        explicit_target = getattr(self.vehicle, "target_lane_index", None)

        if (
            explicit_target is not None
            and current is not None
            and explicit_target != current
            and explicit_target[:2] == current[:2]
        ):
            try:
                self.road.network.get_lane(explicit_target)
                return explicit_target
            except Exception:
                pass

        # Default merge target of the weaving lane.
        if current == ("b", "c", 2):
            return ("b", "c", 1)

        return None

    @staticmethod
    def _bumper_gap(reference, other, centre_distance):
        """Convert centre-to-centre longitudinal distance to bumper gap."""
        half_lengths = 0.5 * (
            float(getattr(reference, "LENGTH", 0.0))
            + float(getattr(other, "LENGTH", 0.0))
        )
        return max(0.0, float(centre_distance) - half_lengths)

    def _lane_neighbours(
        self,
        reference,
        lane_index,
        include_next_front=False,
        include_previous_rear=True,
    ):
        """Find the nearest front/rear vehicles relative to ``reference``.

        The reference vehicle does not need to physically occupy ``lane_index``;
        this is important for evaluating an adjacent target lane. The reference
        is projected onto that lane using ``local_coordinates``.

        In addition to vehicles on ``lane_index`` itself, the immediately
        connected road segments can be considered:

        * ``include_next_front`` includes the closest front vehicle on the next
          lane segment, so front TTC remains continuous across a segment end.
        * ``include_previous_rear`` includes the closest rear vehicle on the
          preceding lane segment, so rear TTC remains continuous across a
          segment start. This is particularly important when the ego is near the
          beginning of the weaving segment while a target-lane follower is still
          on the preceding highway segment.

        The preceding lane is selected from all lanes entering the start node of
        ``lane_index``. Lanes with the same lane ID are preferred; when lane IDs
        change between segments (e.g. the ramp), geometric continuity at the
        segment boundary is used as a fallback.

        Returns
        -------
        front, rear, front_gap, rear_gap
            Vehicle references and bumper-to-bumper longitudinal gaps [m].
        """
        try:
            lane = self.road.network.get_lane(lane_index)
        except Exception:
            return None, None, np.inf, np.inf

        s_ref, _ = lane.local_coordinates(reference.position)

        front = None
        rear = None
        front_gap = np.inf
        rear_gap = np.inf

        # ------------------------------------------------------------------
        # Vehicles on the same lane segment
        # ------------------------------------------------------------------
        for other in self.road.vehicles:
            if other is reference or other.lane_index != lane_index:
                continue

            s_other, _ = lane.local_coordinates(other.position)
            delta_s = float(s_other - s_ref)

            if delta_s >= 0.0:
                gap = self._bumper_gap(reference, other, delta_s)
                if gap < front_gap:
                    front = other
                    front_gap = gap
            else:
                gap = self._bumper_gap(reference, other, -delta_s)
                if gap < rear_gap:
                    rear = other
                    rear_gap = gap

        # ------------------------------------------------------------------
        # Immediately preceding lane segment: rear vehicles
        # ------------------------------------------------------------------
        if include_previous_rear:
            try:
                start_node = lane_index[0]
                target_lane_id = lane_index[2]
                graph = self.road.network.graph

                # Collect every lane on an edge that ends at the start node of
                # the current lane. RoadNetwork stores lanes as
                # graph[from_node][to_node][lane_id].
                predecessor_candidates = []
                for from_node, outgoing in graph.items():
                    if start_node not in outgoing:
                        continue

                    predecessor_lanes = outgoing[start_node]
                    for predecessor_lane_id, predecessor_lane in enumerate(
                        predecessor_lanes
                    ):
                        predecessor_index = (
                            from_node,
                            start_node,
                            predecessor_lane_id,
                        )

                        # Measure geometric continuity between the end of the
                        # predecessor and the start of the current lane.
                        predecessor_end = np.asarray(
                            predecessor_lane.position(predecessor_lane.length, 0),
                            dtype=float,
                        )
                        current_start = np.asarray(
                            lane.position(0, 0), dtype=float
                        )
                        endpoint_distance = float(
                            np.linalg.norm(predecessor_end - current_start)
                        )

                        predecessor_heading = float(
                            predecessor_lane.heading_at(predecessor_lane.length)
                        )
                        current_heading = float(lane.heading_at(0))
                        heading_error = abs(
                            np.arctan2(
                                np.sin(predecessor_heading - current_heading),
                                np.cos(predecessor_heading - current_heading),
                            )
                        )

                        # Prefer continuity of lane numbering when available,
                        # then use endpoint/heading agreement to disambiguate
                        # multiple incoming edges.
                        same_lane_id = predecessor_lane_id == target_lane_id
                        score = endpoint_distance + heading_error
                        predecessor_candidates.append(
                            (
                                not same_lane_id,
                                score,
                                predecessor_index,
                                predecessor_lane,
                            )
                        )

                if predecessor_candidates:
                    predecessor_candidates.sort(key=lambda candidate: candidate[:2])
                    _, _, previous_lane_index, previous_lane = (
                        predecessor_candidates[0]
                    )

                    # Distance from a vehicle on the previous segment to the
                    # reference, measured continuously along the road:
                    #
                    #   previous vehicle -> segment boundary -> reference
                    #
                    distance_from_boundary_to_reference = max(0.0, float(s_ref))

                    for other in self.road.vehicles:
                        if (
                            other is reference
                            or other.lane_index != previous_lane_index
                        ):
                            continue

                        s_other, _ = previous_lane.local_coordinates(other.position)
                        distance_to_boundary = max(
                            0.0,
                            float(previous_lane.length - s_other),
                        )
                        centre_distance = (
                            distance_to_boundary
                            + distance_from_boundary_to_reference
                        )

                        gap = self._bumper_gap(
                            reference,
                            other,
                            centre_distance,
                        )
                        if gap < rear_gap:
                            rear = other
                            rear_gap = gap
            except Exception:
                # If the road graph has no valid predecessor, keep the result
                # from the current lane segment.
                pass

        # ------------------------------------------------------------------
        # Immediately following lane segment: front vehicles
        # ------------------------------------------------------------------
        if include_next_front:
            try:
                next_lane_index = self.road.network.next_lane(
                    lane_index, position=reference.position
                )
                if next_lane_index != lane_index:
                    next_lane = self.road.network.get_lane(next_lane_index)
                    distance_to_lane_end = max(0.0, float(lane.length - s_ref))

                    for other in self.road.vehicles:
                        if other is reference or other.lane_index != next_lane_index:
                            continue

                        s_other, _ = next_lane.local_coordinates(other.position)
                        centre_distance = distance_to_lane_end + max(
                            0.0, float(s_other)
                        )
                        gap = self._bumper_gap(reference, other, centre_distance)
                        if gap < front_gap:
                            front = other
                            front_gap = gap
            except Exception:
                # No unique/valid next lane: the same-segment result is still
                # valid and is kept.
                pass

        return front, rear, front_gap, rear_gap 

    def _longitudinal_speed(self, vehicle, lane_index):
        """Vehicle velocity component along the specified lane [m/s]."""
        if vehicle is None:
            return np.nan

        try:
            lane = self.road.network.get_lane(lane_index)
            s, _ = lane.local_coordinates(vehicle.position)
            lane_heading = lane.heading_at(s)
            return float(vehicle.speed * np.cos(vehicle.heading - lane_heading))
        except Exception:
            # All lanes in this environment have the same forward direction
            # over the weaving section; scalar speed is a safe fallback.
            return float(vehicle.speed)

    def _front_ttc(self, ego, front, gap, lane_index):
        """Constant-velocity TTC to a front vehicle [s]."""
        if front is None or not np.isfinite(gap):
            return np.inf
        if gap <= 0.0:
            return 0.0

        ego_speed = self._longitudinal_speed(ego, lane_index)
        front_speed = self._longitudinal_speed(front, front.lane_index)
        closing_speed = ego_speed - front_speed

        if closing_speed <= 0.0:
            return np.inf
        return float(gap / closing_speed)

    def _rear_ttc(self, ego, rear, gap, lane_index):
        """Constant-velocity TTC for a target-lane vehicle approaching from rear."""
        if rear is None or not np.isfinite(gap):
            return np.inf
        if gap <= 0.0:
            return 0.0

        ego_speed = self._longitudinal_speed(ego, lane_index)
        rear_speed = self._longitudinal_speed(rear, rear.lane_index)
        closing_speed = rear_speed - ego_speed

        if closing_speed <= 0.0:
            return np.inf
        return float(gap / closing_speed)

    @staticmethod
    def _target_rear_induced_braking(ego, target_rear, target_front):
        """Counterfactual braking demand caused by inserting ego into target lane.

        For IDM-like traffic vehicles, compare the rear vehicle's predicted
        acceleration before the ego insertion with its predicted acceleration
        when the ego is used as its new leader:

            induced_braking = max(0, a_without_ego - a_with_ego)

        The returned quantity is therefore a positive braking magnitude in
        m/s^2.  It is ``np.nan`` if no target rear vehicle exists or its
        behavior model does not expose the required acceleration function.
        """
        if target_rear is None or not hasattr(target_rear, "acceleration"):
            return np.nan

        try:
            a_without_ego = target_rear.acceleration(
                ego_vehicle=target_rear,
                front_vehicle=target_front,
            )
            a_with_ego = target_rear.acceleration(
                ego_vehicle=target_rear,
                front_vehicle=ego,
            )
            return max(0.0, float(a_without_ego - a_with_ego))
        except (TypeError, AttributeError, ValueError, FloatingPointError):
            return np.nan


    def _is_successfully_merged(self) -> bool:
        """Return True once the ego is fully aligned with a through lane.

        A successful merge is defined by two conditions:

        1. The ego is assigned to one of the two main through lanes
           (lane IDs 0 or 1) on a main-road segment.
        2. The ego heading is aligned with the local heading of that lane
           within ``merge_heading_tolerance_deg``.

        The main-road segment check explicitly excludes the off-ramp
        (``("c", "o", 0)``), which also has lane ID 0.
        """
        ego = self.vehicle

        if ego is None or self.road is None or ego.lane_index is None:
            return False

        lane_index = ego.lane_index

        # Main highway segments containing the two through lanes. Including
        # ("c", "d") ensures that a merge completed very close to the end of
        # the weaving segment is still recognized after crossing the segment
        # boundary.
        through_segments = {
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
        }

        if lane_index[:2] not in through_segments or lane_index[2] not in (0, 1):
            return False

        try:
            lane = self.road.network.get_lane(lane_index)
            longitudinal, _ = lane.local_coordinates(ego.position)

            # Keep the query inside the lane definition for numerical safety
            # near road-segment boundaries.
            longitudinal = float(np.clip(longitudinal, 0.0, lane.length))
            lane_heading = float(lane.heading_at(longitudinal))
        except Exception:
            return False

        ego_heading = float(getattr(ego, "heading", 0.0))

        # Wrapped signed angular difference in [-pi, pi].
        heading_error = np.arctan2(
            np.sin(ego_heading - lane_heading),
            np.cos(ego_heading - lane_heading),
        )

        tolerance = np.deg2rad(
            float(self.config.get("merge_heading_tolerance_deg", 3.0))
        )

        return bool(abs(heading_error) <= tolerance)

    def _rear_end_collision_by_other_vehicle(self) -> bool:
        """Return True if another vehicle collided with the ego from behind.

        ``highway-env`` exposes a generic ``crashed`` flag but does not, in
        this environment, directly identify which vehicle caused the contact.
        The collision direction is therefore classified geometrically in the
        ego vehicle's local coordinate frame at the state returned by
        ``step()``.

        A collision is classified as a rear-end collision by another vehicle
        only when:

        1. the ego vehicle is marked as crashed;
        2. another vehicle is also marked as crashed and is close enough for
           the two vehicle bounding boxes to overlap (with a small numerical
           tolerance); and
        3. the centre of that vehicle lies behind the ego centre along the
           ego heading.

        This deliberately does *not* use relative speed because some vehicle
        implementations modify their speed immediately when a collision is
        detected.
        """
        ego = self.vehicle

        if (
            ego is None
            or self.road is None
            or not getattr(ego, "crashed", False)
        ):
            return False

        ego_position = np.asarray(ego.position, dtype=float)
        ego_heading = float(getattr(ego, "heading", 0.0))

        # Unit vectors of the ego vehicle's local frame.
        longitudinal_axis = np.array(
            [np.cos(ego_heading), np.sin(ego_heading)], dtype=float
        )
        lateral_axis = np.array(
            [-np.sin(ego_heading), np.cos(ego_heading)], dtype=float
        )

        ego_length = float(getattr(ego, "LENGTH", 5.0))
        ego_width = float(getattr(ego, "WIDTH", 2.0))

        # Small tolerance for discrete simulation steps and floating-point
        # differences in the collision detector.
        longitudinal_tolerance = 0.5
        lateral_tolerance = 0.25

        for other in self.road.vehicles:
            if other is ego:
                continue

            # highway-env marks both vehicles as crashed for a vehicle-vehicle
            # collision. Requiring this avoids confusing an unrelated close
            # follower with the actual collision partner.
            if not getattr(other, "crashed", False):
                continue

            relative_position = (
                np.asarray(other.position, dtype=float) - ego_position
            )

            relative_longitudinal = float(
                np.dot(relative_position, longitudinal_axis)
            )
            relative_lateral = float(
                np.dot(relative_position, lateral_axis)
            )

            other_length = float(getattr(other, "LENGTH", 5.0))
            other_width = float(getattr(other, "WIDTH", 2.0))

            max_longitudinal_separation = (
                0.5 * (ego_length + other_length)
                + longitudinal_tolerance
            )
            max_lateral_separation = (
                0.5 * (ego_width + other_width)
                + lateral_tolerance
            )

            # Approximate bounding-box overlap. At this point ego.crashed is
            # already True, so this test is only used to identify the likely
            # collision partner and its direction.
            in_contact = (
                abs(relative_longitudinal)
                <= max_longitudinal_separation
                and abs(relative_lateral)
                <= max_lateral_separation
            )

            vehicle_is_behind = relative_longitudinal < 0.0

            if in_contact and vehicle_is_behind:
                return True

        return False


    def _agent_reward(self, action: int, vehicle: Vehicle) -> float:
        """
        The vehicle is rewarded for driving with high speed on lanes to the right and avoiding collisions
        But an additional altruistic penalty is also suffered if any vehicle on the merging lane has a low speed.
        :param action: the action performed
        :return: the reward of the state-action transition
        """
        # the optimal reward is 0
        scaled_speed = utils.lmap(
            vehicle.speed, self.config["reward_speed_range"], [0, 1]
        )
        # compute cost for staying on the merging lane
        if vehicle.lane_index == ("b", "c", 2):
            Merging_lane_cost = -np.exp(
                -((vehicle.position[0] - sum(self.ends[:3])) ** 2) / (10 * self.ends[2])
            )
        else:
            Merging_lane_cost = 0

        # give penalty if the agent drives on the offramp
        if vehicle.lane_index == ("c", "o", 0):
            offramp_cost = -self.config["offramp_reward"]
        else:
            offramp_cost = 0

        # lane change cost to avoid unnecessary/frequent lane changes
        Lane_change_cost = (
            -1 * self.config["LANE_CHANGE_COST"] if action == 0 or action == 2 else 0
        )
        # compute headway cost
        headway_distance = self._compute_headway_distance(vehicle)
        Headway_cost = (
            np.log(headway_distance / (self.config["HEADWAY_TIME"] * vehicle.speed))
            if vehicle.speed > 0
            else 0
        )

        # compute overall reward
        reward = (
            self.config["collision_reward"] * (-1 * vehicle.crashed) 
            + (self.config["high_speed_reward"] * np.clip(scaled_speed, 0, 1) )
            + self.config["MERGING_LANE_COST"] * Merging_lane_cost 
            + self.config["HEADWAY_COST"] * (Headway_cost  if Headway_cost < 0 else 0)
            + Lane_change_cost 
            + offramp_cost 
        ) 
        return reward

    def _is_terminal(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        crashes = [veh.crashed for veh in self.road.vehicles]
        return (
            self.vehicle.crashed
            # or self.vehicle.position[0] > 370
            or self.vehicle.position[0] > 310
            or self.vehicle.lane_index == ("c", "o", 0)
            or self.steps > 500
            #or any(crashes)
        )

    def _reset(self) -> None:
        self._make_road()

        if self.config["traffic_density"] == 1:
            # easy mode: 6-8 HDVs
            num_HDV = self.np_random.choice(np.arange(6, 9), 1)[0]
        elif self.config["traffic_density"] == 2:
            # easy mode: 9-12 DVs
            num_HDV = self.np_random.choice(np.arange(9, 13), 1)[0]
        elif self.config["traffic_density"] == 3:
            # easy mode: 13-15 HDVs
            num_HDV = self.np_random.choice(np.arange(13, 16), 1)[0]

        self._make_vehicles(num_HDV)
        self.T = int(self.config["duration"] * self.config["policy_frequency"])

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway-env and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        # use weaving scenario instead of merging
        use_weaving = self.config["use_weaving"]

        # Highway lanes
        self.ends = [150, 80, 80, 150]  # Before, converging, merge, after

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [(c, s), (n, c)]
        line_type_merge = [(c, s), (n, s)]
        for i in range(2):
            net.add_lane(
                "a",
                "b",
                HorizontalLane(
                    [0, y[i]], [sum(self.ends[:2]), y[i]], line_types=line_type[i]
                ),
            )
            net.add_lane(
                "b",
                "c",
                HorizontalLane(
                    [sum(self.ends[:2]), y[i]],
                    [sum(self.ends[:3]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )
            net.add_lane(
                "c",
                "d",
                HorizontalLane(
                    [sum(self.ends[:3]), y[i]],
                    [sum(self.ends[:4]), y[i]],
                    line_types=line_type_merge[i],
                ),
            )

        # Merging lane
        amplitude = 3.25
        ljk = HorizontalLane(
            [0, 6.5 + 4 + 4],
            [self.ends[0], 6.5 + 4 + 4],
            line_types=(c, c),
            forbidden=True,
        )

        lkb = SineLane(
            ljk.position(self.ends[0], -amplitude),
            ljk.position(sum(self.ends[:2]), -amplitude),
            amplitude,
            2 * np.pi / (2 * self.ends[1]),
            np.pi / 2,
            line_types=(c, c),
            forbidden=True,
        )

        lbc = HorizontalLane(
            lkb.position(self.ends[1], 0),
            lkb.position(self.ends[1], 0) + [self.ends[2], 0],
            line_types=(n, c),
            forbidden=False,
        )
        # off ramp
        lco = StraightLane(
            lbc.position(self.ends[2], 0),
            lbc.position(self.ends[2] + 80, 6.5),
            line_types=(c, c),
            forbidden=True,
        )
        lou = HorizontalLane(
            [sum(self.ends[:3]) + 80, 6.5 + 4 + 4],
            [sum(self.ends[:3]) + 80 + 70, 6.5 + 4 + 4],
            line_types=(c, c),
            forbidden=True,
        )

        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)

        if use_weaving:
            # off-ramp
            net.add_lane("c", "o", lco)
            net.add_lane("o", "u", lou)

        road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

        if not use_weaving:
            obstacle = Obstacle(road, lbc.position(self.ends[2], 0))
            obstacle.lane_index = ("b", "c", 2)
            road.objects.append(obstacle)

        self.road = road

    def _make_vehicles(self, num_HDV=3) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        :return: the ego-vehicle
        """
        road = self.road
        other_vehicles_type = utils.class_from_path(self.config["other_vehicles_type"])

        spawn_points_s1 = [10, 50, 90, 130, 170, 210, 225]
        spawn_points_s2 = [0, 40, 80, 120, 160, 200, 220]
        spawn_points_m = [5, 45, 85, 125, 165, 205, 225]
        spawn_points_m_cav = [125, 165]

        # initial speed with noise and location noise
        initial_speed = self.np_random.random(num_HDV + 1) * 8 + 22  # range from [22, 30]
        loc_noise = self.np_random.random(num_HDV + 1) * 6 - 3  # range from [-1.5, 1.5]
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        """Spawn points for CAV"""
        # spawn point indexes on the merging road
        spawn_point_m_c = self.np_random.choice(spawn_points_m_cav, 1, replace=False)
        spawn_point_m_c = list(spawn_point_m_c)
        for c in spawn_point_m_c:
            spawn_points_m.remove(c)
        """spawn the rest CAV on the merging road"""
        ego_vehicle = self.action_type.vehicle_class(
            road,
            road.network.get_lane(("j", "k", 0)).position(
                spawn_point_m_c.pop(0) + loc_noise.pop(0), 0
            ),
            speed=initial_speed.pop(0),
        )
        ego_vehicle.id = 0
        self.vehicle = ego_vehicle
        road.vehicles.append(ego_vehicle)

        self.vehicle.color = (200, 0, 150)

        """Spawn points for HDV"""
        # spawn point indexes on the straight road
        spawn_point_s_h1 = self.np_random.choice(
            spawn_points_s1, num_HDV // 3, replace=False
        )
        spawn_point_s_h2 = self.np_random.choice(
            spawn_points_s2, num_HDV // 3, replace=False
        )
        # spawn point indexes on the merging road
        spawn_point_m_h = self.np_random.choice(
            spawn_points_m, num_HDV - 2 * num_HDV // 3, replace=False
        )
        spawn_point_s_h1 = list(spawn_point_s_h1)
        spawn_point_s_h2 = list(spawn_point_s_h2)
        spawn_point_m_h = list(spawn_point_m_h)

        right_bias = 8.0
        offramp_percentage = 0.3
        biases = list(
            self.np_random.choice(
                [-right_bias, right_bias],
                num_HDV,
                p=[1 - offramp_percentage, offramp_percentage],
            )
        )

        # use weaving scenario instead of merging
        use_weaving = self.config["use_weaving"]

        """spawn the HDV on the main road first"""
        for _ in range(num_HDV // 3):
            veh = other_vehicles_type(
                road,
                road.network.get_lane(("a", "b", 0)).position(
                    spawn_point_s_h1.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
                use_deceleration=use_weaving,
            )

            if use_weaving:
                veh.RIGHT_BIAS = biases.pop(0)
            else:
                veh.RIGHT_BIAS = 0.0

            veh.color = (
                VehicleGraphics.BLUE
                if veh.RIGHT_BIAS == right_bias
                else VehicleGraphics.GREEN
            )
            road.vehicles.append(veh)

        for _ in range(num_HDV // 3):
            veh = other_vehicles_type(
                road,
                road.network.get_lane(("a", "b", 1)).position(
                    spawn_point_s_h2.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
                use_deceleration=use_weaving,
            )

            if use_weaving:
                veh.RIGHT_BIAS = biases.pop(0)
            else:
                veh.RIGHT_BIAS = 0.0

            veh.color = (
                VehicleGraphics.BLUE
                if veh.RIGHT_BIAS == right_bias
                else VehicleGraphics.GREEN
            )
            road.vehicles.append(veh)

        """spawn the rest HDV on the merging road"""
        for _ in range(num_HDV - 2 * num_HDV // 3):
            veh = other_vehicles_type(
                road,
                road.network.get_lane(("j", "k", 0)).position(
                    spawn_point_m_h.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
                use_deceleration=use_weaving,
            )

            # all merging vehicles want on main road (left bias)
            if use_weaving:
                veh.RIGHT_BIAS = -4.0
            else:
                veh.RIGHT_BIAS = 0

            veh.color = VehicleGraphics.GREEN
            road.vehicles.append(veh)


        # # Milano dataset
        # # traj_list = [79, 81, 85, 87,  100, 101, 103, 104, 105, 114, 115, 117, 119, 122, 125, 129, 131, 133, 135, 138, 145, 156, 160]
        # # traj_list = [69, 79, 80, 81, 85, 87, 91, 92, 100, 101, 103, 104, 105, 114, 115, 117, 119, 122, 125, 129, 131]
        # # traj_list = [69, 79, 80, 81, 85, 87, 91, 92, 100, 101, 103, 104, 105, 114, 115, 117, 119, 122, 125, 129, 131, 133, 135, 138, 145, 156, 160]


        # # NGSIM
        # #traj_list = [19, 25, 31, 33, 35, 36, 37, 39, 42, 44, 46, 48, 49, 52, 57, 59, 61, 66, 71, 72, 73, 79, 82, 85, 87, 90, 93, 94, 98, 99]
        # # traj_list = [19, 25, 31, 33, 35, 36, 37, 39, 42, 44, 46, 48, 49, 52, 57, 59, 66, 71, 72, 73, 79, 82, 85, 87, 90, 93, 94, 98, 99]
        # traj_list = [19, 25, 31, 33, 35, 36, 37, 39, 42, 44, 46, 48, 49, 52, 57, 59, 66, 71, 72, 73, 79, 82, 85, 87, 90, 93,  98, 99]
        
        # # start_time = np.random.uniform(5, 12) # Milano dataset
        # start_time = np.random.uniform(57, 65) # NGSIM

        # for traj in traj_list:
        #     v = RealVehicle(f"NGSIM-Dataset/trajectories_numpy/vehicle_{traj}.npy", start_time)
        #     # v = RealVehicle(f"./traj{traj}.npy", start_time)
        #     v.id = traj
        #     v.color = VehicleGraphics.BLACK
        #     road.vehicles.append(v)


register(
    id="merge-single-agent-v0",
    entry_point="highway_env.envs:SingleAgentMergeEnv",
)
