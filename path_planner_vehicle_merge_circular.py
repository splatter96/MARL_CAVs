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
from highway_env.road.objects import Obstacle, ModelObstacle
from highway_env.vehicle.behavior import ModelVehicle, IDMVehicle, ModelIDMVehicle
from highway_env import utils
from highway_env.road.graphics import WorldSurface, RoadGraphics, ModelLaneGraphics
from highway_env.vehicle.graphics import VehicleGraphics

import cProfile, pstats

file_path = "./DEU_MergeCircCont-1_1_T-1.xml"
# file_path = "./DEU_MyTrack2LaneCont-1_1_T-1.xml"
# file_path = "./DEU_HighwayMergeConnected2-1_1_T-1.xml"


class Renderer:
    def __init__(self, road):
        self.road = road

        pygame.init()
        pygame.display.set_caption("Highway-env")  # Also title for i3 config
        # panel_size = (1000, 600)
        # panel_size = (1200, 800)
        panel_size = (1920, 1200)

        self.screen = pygame.display.set_mode([panel_size[0], panel_size[1]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        # self.sim_surface.scaling = 169.0
        # self.sim_surface.scaling = 377.36
        self.sim_surface.scaling = 320.76
        # self.sim_surface.centering_position = [0.7, -0.5]
        self.sim_surface.centering_position = [0.8, -0.7]
        # self.sim_surface.scaling = 4.23
        # self.sim_surface.centering_position = [-0.71, -0.5]

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
            # {"id": 0, "initial_speed": 0.5, "initial_lane_index": 1236},
            {"id": 1, "initial_speed": 0.7, "initial_lane_index": 1234},
            {"id": 2, "initial_speed": 0.5, "initial_lane_index": 1234},
            {"id": 3, "initial_speed": 0.4, "initial_lane_index": 1234},
            {"id": 4, "initial_speed": 0.5, "initial_lane_index": 1235},
            {"id": 5, "initial_speed": 0.7, "initial_lane_index": 1235},
            {"id": 6, "initial_speed": 0.7, "initial_lane_index": 1234},
            {"id": 7, "initial_speed": 0.7, "initial_lane_index": 1235},
            {"id": 8, "initial_speed": 0.7, "initial_lane_index": 1235},
            {"id": 9, "initial_speed": 0.7, "initial_lane_index": 1235},
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
        # profiler = cProfile.Profile()
        # profiler.enable()

        start = time.time()
        for i in range(10000):
            # for i in range(100):
            # print(f"step {i}")
            self.road.act()
            self.road.step(self.timer_period)

            self.renderer.render()
            self.renderer.handle_events()

            # time.sleep(self.timer_period)
            time.sleep(0.01)
            # print()
            # print()

        end = time.time()
        print((end - start) / 10000)
        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.dump_stats("profile_commonroad_views.log")

    def make_road(self):
        scenario, _ = CommonRoadFileReader(file_path).open()

        net = scenario.lanelet_network
        net_common_road = RoadNetworkCommonRoad(net, is_ring=True)

        self.road = RoadCommonRoad(network=net_common_road)
        merge_obstacle = ModelObstacle(self.road, np.array([1.15, -0.6]))
        merge_obstacle.lane_index = 1236
        self.road.objects.append(merge_obstacle)

    def make_vehicles(self):
        # Vehicles on main road
        for car in self.virtual_car_configs:
            id = car["id"]
            v = ModelIDMVehicle.make_on_lane(
                self.road,
                car["initial_lane_index"],
                0.5 * id,
                car["initial_speed"],
            )
            v.id = id
            # v.enable_lane_change = False
            self.road.vehicles.append(v)

        # Vehicles on onramp
        v = ModelIDMVehicle.make_on_lane(
            self.road,
            1236,
            0.1,
            0.5,
        )
        v.id = -1
        self.road.vehicles.append(v)
        v = ModelIDMVehicle.make_on_lane(
            self.road,
            1236,
            0.3,
            0.5,
        )
        v.id = -2
        self.road.vehicles.append(v)
        v = ModelIDMVehicle.make_on_lane(
            self.road,
            1236,
            0.6,
            0.5,
        )
        v.id = -3
        self.road.vehicles.append(v)

    def othermake_vehicles(self):
        """Spawn points for HDV"""
        num_CAV = 1
        num_HDV = 16
        spawn_points_s1 = [10, 50, 90, 130, 170, 210, 225]
        spawn_points_s2 = [0, 40, 80, 120, 160, 200, 220]
        spawn_points_m = [5, 45, 85, 125, 165, 205, 225]

        right_bias = 8.0
        offramp_percentage = 0.3
        biases = list(
            np.random.choice(
                [-right_bias, right_bias],
                num_HDV,
                p=[1 - offramp_percentage, offramp_percentage],
            )
        )

        # initial speed with noise and location noise
        initial_speed = (
            np.random.rand(num_CAV + num_HDV) * 8 + 22
        )  # range from [25, 30]
        loc_noise = np.random.rand(num_CAV + num_HDV) * 6 - 3  # range from [-1.5, 1.5]
        initial_speed = list(initial_speed)
        loc_noise = list(loc_noise)

        # spawn point indexes on the straight road
        spawn_point_s_h1 = np.random.choice(
            spawn_points_s1, num_HDV // 3, replace=False
        )
        spawn_point_s_h2 = np.random.choice(
            spawn_points_s2, num_HDV // 3, replace=False
        )
        # spawn point indexes on the merging road
        spawn_point_m_h = np.random.choice(
            spawn_points_m, num_HDV - 2 * num_HDV // 3, replace=False
        )
        spawn_point_s_h1 = list(spawn_point_s_h1)
        spawn_point_s_h2 = list(spawn_point_s_h2)
        spawn_point_m_h = list(spawn_point_m_h)

        """spawn the HDV on the main road first"""
        i = 0
        for _ in range(num_HDV // 3):
            veh = IDMVehicle(
                self.road,
                self.road.network.get_lane(1).position(
                    spawn_point_s_h1.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
            )
            veh.id = i
            veh.RIGHT_BIAS = biases.pop(0)
            veh.color = (
                VehicleGraphics.BLUE
                if veh.RIGHT_BIAS == right_bias
                else VehicleGraphics.GREEN
            )
            self.road.vehicles.append(veh)
            i += 1

        for _ in range(num_HDV // 3):
            veh = IDMVehicle(
                self.road,
                self.road.network.get_lane(3).position(
                    spawn_point_s_h2.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
            )
            veh.id = i
            veh.RIGHT_BIAS = biases.pop(0)
            veh.color = (
                VehicleGraphics.BLUE
                if veh.RIGHT_BIAS == right_bias
                else VehicleGraphics.GREEN
            )
            self.road.vehicles.append(veh)
            i += 1

        """spawn the rest HDV on the merging road"""
        for _ in range(num_HDV - 2 * num_HDV // 3):
            veh = IDMVehicle(
                self.road,
                self.road.network.get_lane(7146164179188).position(
                    spawn_point_m_h.pop(0) + loc_noise.pop(0), 0
                ),
                speed=initial_speed.pop(0),
            )
            veh.RIGHT_BIAS = -4.0
            veh.color = VehicleGraphics.GREEN
            veh.id = i
            i += 1

            # all merging vehicles want on main road (left bias)
            self.road.vehicles.append(veh)


def main(args=None):
    path_planner = PathPlanner()
    path_planner.start_simulation()


if __name__ == "__main__":
    main()
