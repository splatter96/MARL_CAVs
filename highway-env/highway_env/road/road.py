# cython: profile=True

import numpy as np
from typing import List, Tuple, Dict, TYPE_CHECKING, Optional

from commonroad.scenario.lanelet import LaneletNetwork, Lanelet

from highway_env.road.lane import (
    CommonRoadLane,
)
from highway_env import utils

if TYPE_CHECKING:
    from highway_env.vehicle import kinematics
    from highway_env.road import objects


class RoadNetworkCommonRoad(object):
    lanelet_network: LaneletNetwork

    def __init__(self, net, is_ring=False):
        self.lanelet_network = net

        self.lanes = dict()
        ids = self.lanelet_network._lanelets.keys()

        self.lane_ids = np.array(list(ids), dtype=int)

        # create our custom lane definitions
        for id in ids:
            self.lanes[id] = CommonRoadLane(
                self.lanelet_network.find_lanelet_by_id(id), is_ring
            )

    def get_lane(self, index: int) -> CommonRoadLane:
        """
        Get the lanelet_network corresponding to a given index in the road network.

        :param index: id of the lanelet.
        :return: the corresponding lanelet.
        """
        return self.lanes[index]

    def get_closest_lane_index(self, point: np.ndarray, heading=None) -> int:
        """
        Get the the lane closest to a world position.

        :param point: a world position [m].
        :return: the closest lane.
        """

        # distances for sorting
        distance_list = np.zeros(len(self.lane_ids), dtype=np.float32)

        # go through list of lanelets
        for i, id in enumerate(self.lane_ids):
            lanelet = self.get_lane(id)

            # compute minimum distances to each road
            distance_list[i] = utils.pymindist(
                lanelet.lengths, lanelet.lanelet.center_vertices, point
            )

        # get lanelet with smallest distance
        min_index = distance_list.argmin()
        return self.lane_ids[min_index]

    def next_lane(
        self,
        current_index: int,
        route=None,
        position: np.ndarray = None,
        np_random: np.random.RandomState = np.random,
    ) -> Optional[int]:
        """
        Get the index of the next lane that should be followed after finishing the current lane.
        :param current_index: the index of the current lane.
        :param route: the planned route, if any.
        :param position: the vehicle position.
        :param np_random: a source of randomness.
        :return: the index of the next lane to be followed when current lane is finished.
        """
        # Assumption only one successor
        successors = self.lanelet_network.find_lanelet_by_id(current_index).successor
        if len(successors) > 0:
            return self.lanelet_network.find_lanelet_by_id(current_index).successor[0]
        # if there is no successor road just follow the current on
        return self.lanelet_network.find_lanelet_by_id(current_index).lanelet_id

    def side_lanes(self, lane_index: int) -> List[Lanelet]:
        """
        :param lane_index: the index of a lane.
        :return: indexes of lanes next to a an input lane, to its right or left.
        """
        lanelet = self.lanelet_network.find_lanelet_by_id(lane_index)
        ids = []
        if lanelet.adj_left is not None:
            ids.append(lanelet.adj_left)
        if lanelet.adj_right is not None:
            ids.append(lanelet.adj_right)
        return ids


class RoadCommonRoad(object):
    """A road is a set of lanes, and a set of vehicles driving on these lanes."""

    def __init__(
        self,
        network: RoadNetworkCommonRoad = None,
        vehicles: List["kinematics.Vehicle"] = None,
        road_objects: List["objects.RoadObject"] = None,
        np_random: np.random.RandomState = None,
        record_history: bool = False,
    ) -> None:
        """
        New road.

        :param network: the road network describing the lanes
        :param vehicles: the vehicles driving on the road
        :param road_objects: the objects on the road including obstacles and landmarks
        :param np.random.RandomState np_random: a random number generator for vehicle behaviour
        :param record_history: whether the recent trajectories of vehicles should be recorded for display
        """
        self.network = network
        self.vehicles = vehicles or []
        self.objects = road_objects or []
        self.np_random = np_random if np_random else np.random.RandomState()
        self.record_history = record_history

    def close_vehicles_to(
        self,
        vehicle: "kinematics.Vehicle",
        distance: float,
        count: int = None,
        see_behind: bool = True,
    ) -> object:
        distance = (
            distance**2
        )  # need to square it, because hacky norm does not use sqrt
        vehicles = [
            v
            for v in self.vehicles
            if utils.norm(v.position, vehicle.position) < distance
            and v is not vehicle
            and (see_behind or -2 * vehicle.LENGTH < vehicle.lane_distance_to(v))
        ]

        vehicles = sorted(vehicles, key=lambda v: abs(vehicle.lane_distance_to(v)))
        if count:
            vehicles = vehicles[:count]
        return vehicles

    def act(self) -> None:
        """Decide the actions of each entity on the road."""
        for vehicle in self.vehicles:  # all the vehicles on the road
            vehicle.act()

    def step(self, dt) -> None:
        """
        Step the dynamics of each entity on the road.

        :param dt: timestep [s]
        """
        vehicles = self.vehicles
        objects = self.objects
        len_v = len(vehicles)
        len_o = len(objects)

        for i in range(len_v):
            vehicles[i].step(dt)

        # TODO check collision only every Xth step
        # TODO collect all vehicle positions and check collision at once
        for i in range(len_v):
            v = vehicles[i]
            for j in range(len_v):
                v.check_collision(vehicles[j])
            for j in range(len_o):
                v.check_collision(objects[j])

        # remove vehicles that reached the end of the road
        for v in self.vehicles:
            if self.network.get_lane(v.lane_index).after_end(
                v.position, vehicle_length=v.LENGTH
            ):
                self.vehicles.remove(v)

    def surrounding_vehicles(
        self, vehicle: "kinematics.Vehicle", lane_index: Optional[int] = None
    ) -> Tuple[Optional["kinematics.Vehicle"], Optional["kinematics.Vehicle"]]:
        """
        Find the preceding and following vehicles of a given vehicle.

        :param vehicle: the vehicle whose neighbours must be found
        :param lane_index: the lane on which to look for preceding and following vehicles.
                     It doesn't have to be the current vehicle lane but can also be another lane, in which case the
                     vehicle is projected on it considering its local coordinates in the lane.
        :return: its preceding vehicle, its following vehicle
        """
        lane_index = lane_index or vehicle.lane_index
        if not lane_index:
            return None, None
        lane = self.network.get_lane(lane_index)

        s_front = s_rear = None
        v_front = v_rear = None
        for v in self.vehicles + self.objects:
            if v is not vehicle:  # and not isinstance(v, Landmark):
                if not (
                    v.lane_index == lane_index
                    # or lane_index
                    # == lane.lanelet.adj_right  # check left and right of the lane we want to change to also to avoid to vehicles changing to the same lane and crashing
                    # or lane_index == lane.lanelet.adj_left
                ):
                    continue

                d = lane.distance_between_points(vehicle.position, v.position)

                if d >= 0 and (s_front is None or abs(d) <= s_front):
                    s_front = d
                    v_front = v
                if d < 0 and (s_rear is None or abs(d) < abs(s_rear)):
                    s_rear = d
                    v_rear = v

        return v_front, v_rear
