import gymnasium as gym
import sys
import numpy as np
import warnings
from distutils.dir_util import copy_tree
from shutil import copy
from datetime import datetime
from common.utils import init_dir

import hydra
import omegaconf

import wandb
from wandb.integration.sb3 import WandbCallback

from sb3_contrib import SACD
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    EveryNTimesteps,
    BaseCallback,
    EventCallback,
    CallbackList,
)


warnings.filterwarnings("ignore")

print(sys.path)
if "/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env" in sys.path:
    sys.path.remove("/home/paul/Documents/PhD/RL/MARL_CAVs_lidar/highway-env")
if "/home/paul/Documents/PhD/RL/highway_env_commonroad/highway-env" in sys.path:
    sys.path.remove("/home/paul/Documents/PhD/RL/highway_env_commonroad/highway-env")

sys.path.append("../highway-env")
import highway_env


def init_logging(log_folder, cfg):
    run = wandb.init(
        project=cfg.logging.wandb_project,
        name=log_folder,
        config=omegaconf.OmegaConf.to_container(
            cfg, resolve=True, throw_on_missing=True
        ),
        sync_tensorboard=True,
        save_code=True,
    )

    return run


def init_seeding(seed):
    np.random.seed(seed)


def init_folder(cfg):
    # create an experiment folder
    base_dir = "results/"
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = base_dir + now
    output_dir += "_" + cfg.logging.exp_tag
    output_dir += "_" + str(cfg.seed)
    dirs = init_dir(output_dir, pathes=["configs", "models", "logs", "output"])

    # copy all files to the results that have influence on it
    copy_tree("../highway-env", dirs["configs"])
    copy("configs/configs_sacd.json", dirs["configs"])
    copy(__file__, dirs["configs"])
    with open(dirs["configs"] + "args", "w") as f:
        for arg in sys.argv:
            f.write(f"{arg} ")

    return dirs


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


@hydra.main(version_base="1.1", config_path="./configs", config_name="configs_sacd.yml")
def main(cfg: "DictConfig"):  # noqa: F821
    dirs = init_folder(cfg)
    wandb_run = init_logging(dirs["output"], cfg)

    init_seeding(cfg.seed)

    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    env = make_vec_env("merge-single-agent-v0", n_envs=32, vec_env_cls=SubprocVecEnv)
    old_config = env.get_attr("config")[0]
    old_config.update(config["env"])
    env.env_method("set_config", old_config)

    # for curriculum learning start from difficulty 1
    curriculum = cfg.curriculum

    if curriculum:
        old_config = env.get_attr("config")[0]
        old_config["traffic_density"] = 1
        env.env_method("set_config", old_config)

    model = SACD(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[cfg.algo.net_size, cfg.algo.net_size]),
        learning_starts=cfg.algo.learning_starts,
        buffer_size=cfg.algo.buffer_size,
        learning_rate=cfg.algo.learning_rate,
        train_freq=cfg.algo.train_freq,
        gradient_steps=cfg.algo.gradient_steps,
        max_grad_norm=cfg.algo.max_grad_norm,
        target_update_interval=cfg.algo.target_update_interval,
        batch_size=cfg.algo.batch_size,
        seed=cfg.seed,
        gamma=0.99,
        verbose=1,
        tensorboard_log=dirs["logs"],
        # device=f"cuda:{args.gpu}",
    )

    # configure callbacks
    eval_env = gym.make("merge-single-agent-v0")
    eval_env.config.update(config["env"])
    eval_env.config["traffic_density"] = 3

    custom_eval = CustomEvalCallback(eval_env, dirs["logs"])
    wandb_callback = WandbCallback(verbose=2)
    eval_callback = EveryNTimesteps(n_steps=2500, callback=custom_eval)
    checkpoint_log_speed = TensorboardCallback()

    callback_list = CallbackList([eval_callback, checkpoint_log_speed, wandb_callback])

    # configure logging
    custom_logger = configure(dirs["logs"], ["stdout", "csv", "tensorboard"])
    model.set_logger(custom_logger)

    # split up total learning steps when using curriculum learning
    learn_steps = 20e5
    if curriculum:
        learn_steps = 5e5

    model.learn(
        int(learn_steps),
        tb_log_name=cfg.logging.exp_tag + f"_seed_{cfg.seed}",
        callback=callback_list,
    )

    if curriculum:
        old_config["traffic_density"] = 2
        env.env_method("set_config", old_config)

        model.learn(
            int(5e5),
            tb_log_name=cfg.logging.exp_tag + f"_seed_{cfg.seed}",
            reset_num_timesteps=False,
            callback=callback_list,
        )

        old_config["traffic_density"] = 3
        env.env_method("set_config", old_config)

        model.learn(
            int(10e5),
            tb_log_name=cfg.logging.exp_tag + f"_seed_{cfg.seed}",
            reset_num_timesteps=False,
            callback=callback_list,
        )

    model.save(dirs["models"] + f"/model_{cfg.logging.exp_tag}_seed_{cfg.seed}")

    wandb_run.finish()


if __name__ == "__main__":
    main()
