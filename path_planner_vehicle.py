import sys
import numpy as np
import pygame
import time

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt

sys.path.remove("/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env")
sys.path.append("/home/paul/Documents/PhD/RL/MARL_CAVs_commonroad/highway-env")

from highway_env.road.lane import StraightLane, CircularLane, LineType
from highway_env.road.road import (
    Road,
    RoadNetwork,
    RoadNetworkCommonRoad,
    RoadCommonRoad,
)
from highway_env.vehicle.behavior import ModelVehicle, IDMVehicle, ModelIDMVehicle
from highway_env import utils
from highway_env.road.graphics import WorldSurface, RoadGraphics, ModelLaneGraphics
from highway_env.vehicle.graphics import VehicleGraphics

import cProfile, pstats

# file_path = "./ZAM_Test2Circ-1_1_T-1.xml"
file_path = "./DEU_MyTrack2LaneCont-1_1_T-1.xml"


class Renderer:
    def __init__(self, road):
        self.road = road

        pygame.init()
        pygame.display.set_caption("Highway-env")
        panel_size = (1000, 600)

        self.screen = pygame.display.set_mode([panel_size[0], panel_size[1]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        self.sim_surface.scaling = 169.0
        self.sim_surface.centering_position = [0.7, -0.3]

        """the world position of the center of the displayed window."""
        self.window_position = np.array([2, 0])

    def render(self):
        self.sim_surface.move_display_window_to(self.window_position)

        RoadGraphics.display(self.road, self.sim_surface, ModelLaneGraphics)

        RoadGraphics.display_road_objects(self.road, self.sim_surface, offscreen=False)

        RoadGraphics.display_traffic(self.road, self.sim_surface, offscreen=False)

        self.screen.blit(self.sim_surface, (0, 0))
        pygame.display.flip()

    def handle_events(self) -> None:
        """Handle pygame events by forwarding them to the display and environment vehicle."""
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                paused = True
                while paused:
                    pygame.time.wait(1000)
                    for event in pygame.event.get():
                        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                            paused = False
            elif event.type == pygame.MOUSEBUTTONUP:
                pix_pos = pygame.Vector2(event.pos)
                # print(pix_pos)
                # print(self.sim_surface.pix2pos(pix_pos[0], pix_pos[1]))
                # print(self.road.vehicles[0].position)
                self.road.vehicles[0].position = np.array(
                    [*self.sim_surface.pix2pos(pix_pos[0], pix_pos[1])]
                )

            self.sim_surface.handle_event(event)


class PathPlanner:
    def __init__(self):
        self.virtual_car_configs = [
            # {"id": 0, "initial_speed": 0.9, "initial_lane_index": 1234},
            # {"id": 1, "initial_speed": 0.9, "initial_lane_index": 1234},
            # {"id": 2, "initial_speed": 0.3, "initial_lane_index": 1234},
            # {"id": 3, "initial_speed": 0.8, "initial_lane_index": 5678},
            # {"id": 4, "initial_speed": 0.7, "initial_lane_index": 5678},
            # {"id": 5, "initial_speed": 0.9, "initial_lane_index": 5678},
            {"id": 0, "initial_speed": 0.7, "initial_lane_index": 1234},
            {"id": 1, "initial_speed": 0.5, "initial_lane_index": 1234},
            {"id": 2, "initial_speed": 0.4, "initial_lane_index": 1234},
            {"id": 3, "initial_speed": 0.5, "initial_lane_index": 5678},
            {"id": 4, "initial_speed": 0.7, "initial_lane_index": 5678},
            {"id": 5, "initial_speed": 0.5, "initial_lane_index": 5678},
        ]

        # TODO extend to be able to hold multiple real vehicles
        self.heading_offset = 0.0  # [rad]
        self.steering_offset = 0.0  # [deg]

        self.timer_period = 0.05  # [s]

        self.make_road()
        self.make_vehicles()

        self.renderer = Renderer(self.road)

        self.subs = []

    def start_simulation(self):
        profiler = cProfile.Profile()
        profiler.enable()

        start = time.time()
        # for i in range(10000):
        for i in range(100):
            # print(f"step {i}")
            self.road.act()
            self.road.step(self.timer_period)

            # self.renderer.render()
            # self.renderer.handle_events()

            # time.sleep(self.timer_period)
            # time.sleep(0.01)

        end = time.time()
        print((end - start) / 10000)
        profiler.disable()
        stats = pstats.Stats(profiler)
        stats.dump_stats("profile_commonroad_no_list.log")

    def make_road(self):
        # lane = StraightLane([0, 0], [3.5, 0], line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE), width=0.3)
        # lane_o = StraightLane([0, -0.3], [3.5, -0.3], line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE), width=0.3)

        # lane2 = CircularLane(center=[3.4, 1], radius=1, start_phase=-np.pi/2, end_phase=np.pi/2, width=0.3)
        # lane2_o = CircularLane(center=[3.4, 1], radius=1.3, start_phase=-np.pi/2, end_phase=np.pi/2, width=0.3)

        # lane3 = StraightLane([3.5, 2.0], [0, 2.0], line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE), width=0.3)
        # lane3_o = StraightLane([3.5, 2.3], [0.0, 2.3], line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE), width=0.3)

        # lane4 = CircularLane(center=[0.0, 1.0], radius=1.0, start_phase=np.pi/2, end_phase=1.5*np.pi, width=0.3)
        # lane4_o = CircularLane(center=[0.0, 1.0], radius=1.3, start_phase=np.pi/2, end_phase=1.5*np.pi, width=0.3)
        #

        scenario, planning_problem_set = CommonRoadFileReader(file_path).open()

        net = scenario.lanelet_network
        print(len(net.lanelets))

        net_common_road = RoadNetworkCommonRoad(net)

        print(net_common_road.get_closest_lane_index(np.array([0, 0])))
        print(net_common_road.next_lane(1234))
        print(f"Sidelanes: {net_common_road.side_lanes(1234)}")

        l = net_common_road.get_lane(1234)
        print(l)

        point = l.local_coordinates(np.array([-0.1, -0.0]))
        # point = l.local_coordinates(np.array([1.7, -1.0]))
        print(point)

        # test_point = l.position(2.56, 2.0)
        # print(f"testpoint {test_point}")

        # local_point = l.position(1.8, -0.1)
        local_point = l.position(point[0], point[1])
        print(local_point)

        # right_corner = l.local_coordinates(np.array([1.7, -1]))
        # heading = l.heading_at(point[0])
        heading = l.heading_at(7.6)
        print(np.rad2deg(heading))

        # exit(0)

        straight_length = 0.8

        lane = StraightLane(
            [0, 0],
            [straight_length, 0],
            line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE),
            width=0.3,
        )
        lane_o = StraightLane(
            [0, -0.3],
            [straight_length, -0.3],
            line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE),
            width=0.3,
        )

        lane2 = CircularLane(
            center=[straight_length - 0.1, 1],
            radius=1,
            start_phase=-np.pi / 2,
            end_phase=np.pi / 2,
            width=0.3,
        )
        lane2_o = CircularLane(
            center=[straight_length - 0.1, 1],
            radius=1.3,
            start_phase=-np.pi / 2,
            end_phase=np.pi / 2,
            width=0.3,
        )

        lane3 = StraightLane(
            [straight_length, 2.0],
            [0, 2.0],
            line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE),
            width=0.3,
        )
        lane3_o = StraightLane(
            [straight_length, 2.3],
            [0.0, 2.3],
            line_types=(LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE),
            width=0.3,
        )

        lane4 = CircularLane(
            center=[0.0, 1.0],
            radius=1.0,
            start_phase=np.pi / 2,
            end_phase=1.5 * np.pi,
            width=0.3,
        )
        lane4_o = CircularLane(
            center=[0.0, 1.0],
            radius=1.3,
            start_phase=np.pi / 2,
            end_phase=1.5 * np.pi,
            width=0.3,
        )

        net = RoadNetwork()
        net.add_lane("a", "b", lane)
        net.add_lane("a", "b", lane_o)

        net.add_lane("b", "c", lane2)
        net.add_lane("b", "c", lane2_o)

        net.add_lane("c", "d", lane3)
        net.add_lane("c", "d", lane3_o)

        net.add_lane("d", "a", lane4)
        net.add_lane("d", "a", lane4_o)

        # print(net.side_lanes(("a", "b", 1)))

        # self.road = Road(network=net)
        self.road = RoadCommonRoad(network=net_common_road)

    def make_vehicles(self):
        for car in self.virtual_car_configs:
            id = car["id"]
            # v = ModelIDMVehicle(
            #     self.road,
            #     np.array([0.5 * id, 0], dtype=np.float64),
            #     speed=car["initial_speed"],
            # )
            v = ModelIDMVehicle.make_on_lane(
                self.road, car["initial_lane_index"], 0.5 * id, car["initial_speed"]
            )
            v.id = id
            # v.enable_lane_change = False
            self.road.vehicles.append(v)


def main(args=None):
    path_planner = PathPlanner()
    path_planner.start_simulation()


if __name__ == "__main__":
    main()
