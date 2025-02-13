import sys
import numpy as np
import pygame
import time
from copy import deepcopy

from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.visualization.mp_renderer import MPRenderer
import matplotlib.pyplot as plt

sys.path.remove("/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env")
# sys.path.append("/home/paul/Documents/PhD/RL/MARL_CAVs_commonroad/highway-env")
sys.path.append("/home/paul/Documents/PhD/RL/MARL_CAVs_commonroad/highway-env_old")

from highway_env.road.lane import (
    StraightLane,
    HorizontalLane,
    SineLane,
    CircularLane,
    LineType,
    DEFAULT_WIDTH,
)
from highway_env.road.road import (
    Road,
    RoadNetwork,
    # RoadNetworkCommonRoad,
    # RoadCommonRoad,
)
from highway_env.road.objects import Obstacle
from highway_env.vehicle.behavior import IDMVehicle
from highway_env import utils
from highway_env.road.graphics import WorldSurface, RoadGraphics  # , ModelLaneGraphics
from highway_env.vehicle.graphics import VehicleGraphics

import cProfile, pstats

# file_path = "./DEU_MyTrack2LaneCont-1_1_T-1.xml"
# file_path = "./DEU_HighwayMergeConnected2-1_1_T-1.xml"
file_path = "./DEU_HighwayMergeConnected2longer-1_1_T-1.xml"

import pickle


class Renderer:
    def __init__(self, road):
        self.road = road

        pygame.init()
        pygame.display.set_caption("Highway-env")
        # panel_size = (1000, 600)
        panel_size = (1200, 800)

        self.screen = pygame.display.set_mode([panel_size[0], panel_size[1]])
        self.sim_surface = WorldSurface(panel_size, 0, pygame.Surface(panel_size))
        # self.sim_surface.scaling = 169.0
        # self.sim_surface.centering_position = [0.7, -0.3]
        self.sim_surface.scaling = 5.5
        self.sim_surface.centering_position = [-0.61, -0.6]
        # self.sim_surface.centering_position = [-0.61, 0.5]

        """the world position of the center of the displayed window."""
        self.window_position = np.array([2, 0])

    def render(self):
        self.sim_surface.move_display_window_to(self.window_position)

        RoadGraphics.display(self.road, self.sim_surface)  # , ModelLaneGraphics)

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
            # {"id": 0, "initial_speed": 0.7, "initial_lane_index": 1234},
            # {"id": 1, "initial_speed": 0.5, "initial_lane_index": 1234},
            # {"id": 2, "initial_speed": 0.4, "initial_lane_index": 1234},
            # {"id": 3, "initial_speed": 0.5, "initial_lane_index": 5678},
            # {"id": 4, "initial_speed": 0.7, "initial_lane_index": 5678},
            # {"id": 5, "initial_speed": 0.5, "initial_lane_index": 5678},
            {"id": 0, "initial_speed": 25.0, "initial_lane_index": 7146164179188},
            {"id": 1, "initial_speed": 26.0, "initial_lane_index": 1},
            {"id": 2, "initial_speed": 25.7, "initial_lane_index": 7146164179188},
            {"id": 3, "initial_speed": 26.5, "initial_lane_index": 1},
            {"id": 4, "initial_speed": 26.7, "initial_lane_index": 7146164179188},
            {"id": 5, "initial_speed": 26.5, "initial_lane_index": 7146164179188},
        ]

        # TODO extend to be able to hold multiple real vehicles
        self.heading_offset = 0.0  # [rad]
        self.steering_offset = 0.0  # [deg]

        self.timer_period = 0.05  # [s]

        self.renderer = None

        # Consistent seeding for evaluation
        seed_ = 21
        np.random.seed(seed_)

        self.make_road_original()
        # self.make_road()
        self.othermake_vehicles()

        self.initial_vehicles = deepcopy(self.road.vehicles)

        # load vehicles from file
        # with open("initial_veh108.pkl", "rb") as f:  # open a text file
        #     self.road.vehicles = pickle.load(f)
        #
        # self.renderer = Renderer(self.road)

        self.subs = []

    def reset(self):
        self.road.vehicles = []
        self.othermake_vehicles()
        self.initial_vehicles = deepcopy(self.road.vehicles)

        # with open("initial_veh108.pkl", "rb") as f:  # open a text file
        #     self.road.vehicles = pickle.load(f)

    def start_simulation(self):
        # profiler = cProfile.Profile()
        # profiler.enable()
        #
        num_runs = 100000
        start = time.time()
        num_crashes = 0
        num_episodes = 0
        average_speed = 0
        for i in range(num_runs):
            # reset simulation in case of crash
            for i in range(len(self.road.vehicles)):
                if self.road.vehicles[i].crashed:
                    num_crashes += 1
                    num_episodes += 1
                    # with open(
                    #     f"initial_veh{num_episodes}.pkl", "wb"
                    # ) as f:  # open a text file
                    #     pickle.dump(self.initial_vehicles, f)  # serialize the list
                    self.reset()
                    break

            # reset simulation when all vehicles reached the end
            if len(self.road.vehicles) == 0:
                num_episodes += 1
                self.reset()
            # vehicles_at_end = 0
            # for i in range(len(self.road.vehicles)):
            #     if self.road.vehicles[i].position[0] > 500:
            #         vehicles_at_end += 1
            #
            # if vehicles_at_end == len(self.road.vehicles):
            #     num_episodes += 1
            #     self.reset()
            #
            self.road.act()
            self.road.step(self.timer_period)

            speed_in_step = 0
            for v in self.road.vehicles:
                speed_in_step += v.speed

            if len(self.road.vehicles) > 0:
                average_speed += speed_in_step / len(self.road.vehicles)
                # print(speed_in_step / len(self.road.vehicles))

            if self.renderer is not None:
                self.renderer.render()
                self.renderer.handle_events()

                time.sleep(0.01)

        end = time.time()
        print((end - start) / num_runs)
        print(f"Crashes {num_crashes}")
        print(f"Episodes {num_episodes}")
        print(f"average_speed {average_speed/num_runs}")
        # profiler.disable()
        # stats = pstats.Stats(profiler)
        # stats.dump_stats("profile_commonroad_no_ring.log")

    def make_road(self):
        scenario, _ = CommonRoadFileReader(file_path).open()

        net = scenario.lanelet_network
        net_common_road = RoadNetworkCommonRoad(net, is_ring=False)

        self.road = RoadCommonRoad(network=net_common_road)
        merge_obstacle = Obstacle(self.road, np.array([310, -4]))
        merge_obstacle.lane_index = 7146164179188
        # self.road.objects.append(merge_obstacle)

    def make_road_original(self) -> None:
        """
        Make a road composed of a straight highway-env and a merging lane.

        :return: the road
        """

        net = RoadNetwork()

        # Highway lanes
        # self.ends = [150, 80, 200, 150]  # Before, converging, merge, after
        self.ends = [150, 80, 80, 150]  # Before, converging, merge, after
        # # self.ends = [150, 80, 40, 40, 150]  # Before, converging, merge, after

        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, DEFAULT_WIDTH]
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
        net.add_lane("c", "o", lco)
        net.add_lane("o", "u", lou)  # off ramp
        road = Road(
            network=net,
            # np_random=self.np_random,
            # record_history=self.config["show_trajectories"],
        )
        self.road = road

    def make_vehicles(self):
        for car in self.virtual_car_configs:
            id = car["id"]
            v = IDMVehicle.make_on_lane(
                self.road,
                car["initial_lane_index"],
                20 * id,
                car["initial_speed"],
            )
            v.id = id
            # v.enable_lane_change = False
            self.road.vehicles.append(v)

    def othermake_vehicles(self):
        """Spawn points for HDV"""
        num_CAV = 1
        num_HDV = 16
        spawn_points_s1 = [10, 50, 90, 130, 170, 210, 225]
        spawn_points_s2 = [0, 40, 80, 120, 160, 200, 220]
        spawn_points_m = [5, 45, 85, 125, 165, 205, 225]

        # merging_lane_index = 7146164179188
        # through_lane1_index = 1
        # through_lane2_index = 3

        merging_lane_index = ("j", "k", 0)
        through_lane1_index = ("a", "b", 0)
        through_lane2_index = ("a", "b", 1)

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
                self.road.network.get_lane(through_lane1_index).position(
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
                self.road.network.get_lane(through_lane2_index).position(
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
                self.road.network.get_lane(merging_lane_index).position(
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
