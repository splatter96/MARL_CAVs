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
    DELTA_SPEED = 5  # [m/s]

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

    def act(self, action: Union[dict, str] = None) -> None:
        """
        Perform a high-level action to change the desired lane or speed.

        - If a high-level action is provided, update the target speed and lane;
        - then, perform longitudinal and lateral control.

        :param action: a high-level action
        """
        self.follow_road()
        if action == "FASTER":
            self.target_speed += self.DELTA_SPEED
        elif action == "SLOWER":
            self.target_speed -= self.DELTA_SPEED
        elif action == "LANE_RIGHT":
            if self.road.network.get_lane(self.target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = self.target_lane_index
        elif action == "LANE_LEFT":
            if self.road.network.get_lane(self.target_lane_index).is_reachable_from(
                self.position
            ):
                self.target_lane_index = self.target_lane_index

        action = {
            "steering": self.steering_control(self.target_lane_index),
            "acceleration": self.speed_control(self.target_speed),
        }
        action["steering"] = np.clip(
            action["steering"], -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE
        )
        super().act(action)

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

        # cdef float speed = self.speed

        # cdef float TAU = self.PURSUIT_TAU
        # cdef float KP = self.KP_LATERAL

        # target_lane = self.road.network.get_lane(target_lane_index)
        # lane_coords = target_lane.local_coordinates(self.position)

        # cdef float lane_x = lane_coords[0]
        # cdef float lane_y = lane_coords[1]

        # lane_next_coords = lane_x + speed * TAU

        # cdef float lane_future_heading = target_lane.heading_at(lane_next_coords)

        # cdef float lateral_speed_command, heading_command, headin_ref, heading_rate_command, steering_angle

        # # Lateral position control
        # lateral_speed_command = -KP * lane_y
        # # Lateral speed to heading
        # heading_command = asin(utils.c_clip(lateral_speed_command / utils.c_not_zero(speed), -1, 1))
        # heading_ref = lane_future_heading + utils.c_clip(heading_command, -np.pi/4, np.pi/4)
        # # Heading control
        # heading_rate_command = self.KP_HEADING * utils.wrap_to_pi(heading_ref - self.heading)
        # # Heading rate to steering angle
        # steering_angle = asin(utils.c_clip(self.LENGTH / 2 / utils.c_not_zero(speed) * heading_rate_command,
        # -1, 1))
        # steering_angle = utils.c_clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        # return float(steering_angle)

    def speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle.
        Using a simple proportional controller.

        :param target_speed: the desired speed
        :return: an acceleration command [m/s2]
        """
        return self.KP_A * (target_speed - self.speed)


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
