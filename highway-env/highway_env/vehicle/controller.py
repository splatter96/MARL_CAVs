# cython: profile=True
from typing import List, Tuple, Union

import numpy as np
from highway_env import utils
from highway_env.types import Vector
from highway_env.vehicle.kinematics import Vehicle


class ControlledVehicle(Vehicle):
    """
    A vehicle piloted by two low-level controller, allowing high-level actions such as cruise control and lane changes.

    - The longitudinal controller is a speed controller;
    - The lateral controller is a heading controller cascaded with a lateral position controller.
    """

    target_speed: float
    """ Desired velocity."""

    TAU_A = 0.6  # [s]
    TAU_DS = 0.2  # [s]
    PURSUIT_TAU = 0.5 * TAU_DS  # [s]
    KP_A = 1 / TAU_A
    KP_HEADING = 1 / TAU_DS
    KP_LATERAL = 1 / 3 * KP_HEADING  # [1/s]
    MAX_STEERING_ANGLE = np.pi / 3  # [rad]

    def __init__(
        self,
        road,
        position: Vector,
        heading: float = 0,
        speed: float = 0,
        target_lane_index=None,
        target_speed: float = None,
        route=None,
    ):
        super().__init__(road, position, heading, speed)
        self.target_lane_index = target_lane_index or self.lane_index
        self.target_speed = target_speed or self.speed

        # for adjustement of speed near intersection
        self.alpha_v0 = 1
        self.route = route

    def follow_road(self) -> bool:
        """At the end of a lane, automatically switch to a next one."""
        if self.road.network.get_lane(self.lane_index).after_end(
            self.position, vehicle_length=self.LENGTH
        ):
            self.target_lane_index = self.road.network.next_lane(
                self.lane_index,
                route=self.route,
                position=self.position,
                np_random=self.road.np_random,
            )
            return True
        return False

    def steering_control(self, target_lane_index) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """

        target_lane = self.road.network.get_lane(target_lane_index)
        return utils.steering_control(
            self.speed,
            self.position,
            self.heading,
            target_lane,
            self.PURSUIT_TAU,
            self.KP_LATERAL,
            self.KP_HEADING,
            self.LENGTH,
            self.MAX_STEERING_ANGLE,
        )


class MDPVehicle(ControlledVehicle):
    """A controlled vehicle with a specified discrete range of allowed target speeds."""

    SPEED_COUNT: int = 6  # [], original = 3
    SPEED_MIN: float = 5  # [m/s]
    SPEED_MAX: float = 30  # [m/s]

    def __init__(
        self,
        road,
        position: np.ndarray,
        heading: float = 0,
        speed: float = 0,
        target_lane_index=None,
        target_speed: float = None,
        route=None,
    ) -> None:
        super().__init__(
            road, position, heading, speed, target_lane_index, target_speed, route
        )
        self.speed_index = self.speed_to_index(self.target_speed)
        self.target_speed = self.index_to_speed(self.speed_index)

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action.

        - If the action is a speed change, choose speed from the allowed discrete range.
        - Else, forward action to the ControlledVehicle handler.

        :param action: a high-level action
        """
        if action == "FASTER":
            self.speed_index = self.speed_to_index(self.speed) + 1
        elif action == "SLOWER":
            self.speed_index = self.speed_to_index(self.speed) - 1
        else:
            super().act(action)
            return
        self.speed_index = int(np.clip(self.speed_index, 0, self.SPEED_COUNT - 1))
        self.target_speed = self.index_to_speed(self.speed_index)
        super().act()

    def index_to_speed(self, index: int) -> float:
        """
        Convert an index among allowed speeds to its corresponding speed

        :param index: the speed index []
        :return: the corresponding speed [m/s]
        """
        if self.SPEED_COUNT > 1:
            return self.SPEED_MIN + index * (self.SPEED_MAX - self.SPEED_MIN) / (
                self.SPEED_COUNT - 1
            )
        else:
            return self.SPEED_MIN

    def speed_to_index(self, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - self.SPEED_MIN) / (self.SPEED_MAX - self.SPEED_MIN)
        return int(
            np.clip(np.round(x * (self.SPEED_COUNT - 1)), 0, self.SPEED_COUNT - 1)
        )

    @classmethod
    def speed_to_index_default(cls, speed: float) -> int:
        """
        Find the index of the closest speed allowed to a given speed.

        :param speed: an input speed [m/s]
        :return: the index of the closest speed allowed []
        """
        x = (speed - cls.SPEED_MIN) / (cls.SPEED_MAX - cls.SPEED_MIN)
        return np.int(
            np.clip(np.round(x * (cls.SPEED_COUNT - 1)), 0, cls.SPEED_COUNT - 1)
        )

    @classmethod
    def get_speed_index(cls, vehicle: Vehicle) -> int:
        return getattr(
            vehicle, "speed_index", cls.speed_to_index_default(vehicle.speed)
        )
