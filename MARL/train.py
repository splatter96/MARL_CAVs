import time
import gymnasium as gym
import sys
import numpy as np
import warnings
import argparse
import json
from distutils.dir_util import copy_tree
from shutil import copy
from datetime import datetime
import os
from common.utils import agg_double_list, copy_file_ppo, init_dir

warnings.filterwarnings("ignore")

if "/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env" in sys.path:
    sys.path.remove("/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env")
sys.path.append("../highway-env")
import highway_env
from sb3_contrib import SACD
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EveryNTimesteps,
    BaseCallback,
    EventCallback,
    EvalCallback,
    CallbackList,
    EveryNTimesteps,
)
from stable_baselines3.common.logger import configure

import wandb
from wandb.integration.sb3 import WandbCallback

import hydra
import omegaconf


class CustomEvalCallback(EventCallback):
    """
    Custom callback to evaluate a policy and save some important metrics
    """

    def __init__(self, env, log_dir, episodes=4, verbose=0):
        super().__init__(verbose=verbose)

        self.env = env
        self.log_dir = log_dir
        self.episodes = episodes

        with open(self.log_dir + "/evaluation.csv", "w") as f:
            f.write("steps,reward,ego_speed,network_speed\n")

    def _on_step(self):
        ret = speed = road_speed = total_steps = 0

        for i in range(self.episodes):
            obs, info = self.env.reset()
            done = truncated = False
            while not (done or truncated):
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = self.env.step(action)

                ret += reward

                speed += info["average_speed"]
                road_speed += info["average_road_speed"]
                total_steps += 1

        f = open(self.log_dir + "/evaluation.csv", "a")
        f.write(
            f"{self.num_timesteps},{ret/self.episodes},{speed/total_steps},{road_speed/total_steps}\n"
        )
        f.close()

        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ego_speed = 0
        self.network_speed = 0
        self.steps = 0

    def _on_step(self) -> bool:
        # Log additional tensor
        # print(self.locals)
        info = self.locals["infos"][0]
        self.ego_speed += info["average_speed"]
        self.network_speed += info["average_road_speed"]
        self.steps += 1

        if self.locals["dones"][0]:
            self.logger.record("ego_speed", self.ego_speed / self.steps)
            self.logger.record("network_speed", self.network_speed / self.steps)

            self.ego_speed = 0
            self.network_speed = 0
            self.steps = 0

        return True


def parse_args():
    default_base_dir = "./results/"
    default_config_dir = "configs/configs_sacd.json"
    parser = argparse.ArgumentParser(
        description=("Train or evaluate policy on RL environment " "using sacd")
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=False,
        default=default_base_dir,
        help="experiment base dir",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        required=False,
        default=default_config_dir,
        help="experiment config path",
    )
    parser.add_argument(
        "--exp-tag",
        type=str,
        required=False,
        default="",
        help="experiment tag to identify experiments",
    )
    parser.add_argument(
        "--gpu", type=int, required=False, default=0, help="index of the GPU to run on"
    )
    parser.add_argument(
        "--seed", type=int, required=False, default=0, help="Overwrite seed in config"
    )
    args = parser.parse_args()
    return args


@hydra.main(version_base="1.1", config_path="./configs", config_name="configs_sacd.yml")
def main(cfg: "DictConfig"):  # noqa: F821
    base_dir = cfg.logging.output_dir + "/"

    # create an experiment folder
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = base_dir + now
    if cfg.logging.exp_tag != "":
        output_dir += "_" + cfg.logging.exp_tag
    if cfg.seed != "":
        output_dir += "_" + str(cfg.seed)
    dirs = init_dir(output_dir, pathes=["configs", "models", "logs", "output"])

    # copy all files to the results that have influence on it
    copy_tree("../highway-env", dirs["configs"])
    copy("configs/configs_sacd.json", dirs["configs"])
    copy(__file__, dirs["configs"])
    with open(dirs["configs"] + "args", "w") as f:
        for arg in sys.argv:
            f.write(f"{arg} ")

    # configure environment
    env_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )["env"]
    env = make_vec_env(
        "merge-single-agent-v0", n_envs=cfg.algo.num_envs, vec_env_cls=SubprocVecEnv
    )
    old_config = env.get_attr("config")[0]
    old_config.update(env_config)
    env.env_method("set_config", old_config)

    # for curriculum learning start from difficulty 1
    curriculum_learning = cfg.curriculum
    if curriculum_learning == True:
        old_config = env.get_attr("config")[0]
        old_config["traffic_density"] = 1
        env.env_method("set_config", old_config)

    seed_ = cfg.seed

    # configure callbacks
    eval_env = gym.make("merge-single-agent-v0")
    eval_env.config.update(env_config)
    eval_env.config["traffic_density"] = 3
    # eval_callback = EvalCallback(eval_env, log_path=dirs['logs'], eval_freq=500, deterministic=True, render=False)
    custom_eval = CustomEvalCallback(eval_env, dirs["logs"])

    eval_callback = EveryNTimesteps(n_steps=500, callback=custom_eval)
    checkpoint_log_speed = TensorboardCallback()

    model = SACD(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[cfg.algo.net_size, cfg.algo.net_size]),
        learning_starts=cfg.algo.learning_starts,
        buffer_size=cfg.algo.learning_starts,
        learning_rate=cfg.algo.learning_rate,
        train_freq=cfg.algo.train_freq,
        gradient_steps=cfg.algo.gradient_steps,
        max_grad_norm=cfg.algo.max_grad_norm,
        target_update_interval=cfg.algo.target_update_interval,
        batch_size=cfg.algo.batch_size,
        gamma=cfg.algo.gamma,
        verbose=1,
        tensorboard_log=dirs["logs"],
        device=f"cuda:{cfg.gpu}",
    )

    run = wandb.init(
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        project=cfg.logging.wandb_project,
        sync_tensorboard=True,
        name=cfg.logging.exp_tag,
    )

    wandb_callback = WandbCallback(model_save_path=f"models/{run.id}", verbose=2)

    callback_list = CallbackList([eval_callback, checkpoint_log_speed, wandb_callback])

    # configure logging
    custom_logger = configure(dirs["logs"], ["stdout", "csv", "tensorboard"])
    model.set_logger(custom_logger)

    # split up total learning steps when using curriculum learning
    learn_steps = 10e5
    if curriculum_learning == True:
        learn_steps = 3e5

    model.learn(
        int(learn_steps),
        tb_log_name=cfg.logging.exp_tag + f"_seed_{seed_}",
        callback=callback_list,
    )

    if curriculum_learning == True:
        old_config["traffic_density"] = 2
        env.env_method("set_config", old_config)

        model.learn(
            int(3e5),
            tb_log_name=cfg.logging.exp_tag + f"_seed_{seed_}",
            reset_num_timesteps=False,
            callback=callback_list,
        )
        old_config["traffic_density"] = 3
        env.env_method("set_config", old_config)

        model.learn(
            int(4e5),
            tb_log_name=cfg.logging.exp_tag + f"_seed_{seed_}",
            reset_num_timesteps=False,
            callback=callback_list,
        )

    model.save(dirs["models"] + f"/model_{cfg.logging.exp_tag}_seed_{seed_}")


if __name__ == "__main__":
    main()
