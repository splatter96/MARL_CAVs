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

warnings.filterwarnings('ignore')

sys.path.append("../highway-env")
import highway_env
from sb3_contrib import SACD
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EveryNTimesteps, BaseCallback, EventCallback, EvalCallback, CallbackList, EveryNTimesteps
from stable_baselines3.common.logger import configure

class CustomEvalCallback(EventCallback):

    def __init__(self, env, log_dir, episodes=4, verbose=0):
        super().__init__(verbose=verbose)

        self.env = env
        self.log_dir = log_dir
        self.episodes = episodes

    def _on_step(self):
        print(f"At {self.num_timesteps}")
        for i in range(episodes):


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
        #print(self.locals)
        info = self.locals["infos"][0]
        self.ego_speed += info["average_speed"]
        self.network_speed += info["average_road_speed"]
        self.steps += 1

        if self.locals["dones"][0]:
            self.logger.record("ego_speed", self.ego_speed/self.steps)
            self.logger.record("network_speed", self.network_speed/self.steps)

            self.ego_speed = 0
            self.network_speed = 0
            self.steps = 0

        return True

def parse_args():
    default_base_dir = "./results/"
    default_config_dir = 'configs/configs_sacd.json'
    parser = argparse.ArgumentParser(description=('Train or evaluate policy on RL environment '
                                                  'using sacd'))
    parser.add_argument('--base-dir', type=str, required=False,
                        default=default_base_dir, help="experiment base dir")
    parser.add_argument('--config-dir', type=str, required=False,
                        default=default_config_dir, help="experiment config path")
    parser.add_argument('--exp-tag', type=str, required=False,
                        default='', help="experiment tag to identify experiments")
    parser.add_argument('--gpu', type=int, required=False,
                        default=0, help="index of the GPU to run on")
    parser.add_argument('--seed', type=int, required=False,
                        default=0, help="Overwrite seed in config")
    args = parser.parse_args()
    return args

def train(args):
    base_dir = args.base_dir
    config_dir = args.config_dir

    # load json config
    with open(config_dir) as f:
        config = json.load(f)

    # create an experiment folder
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = base_dir + now
    if args.exp_tag != '':
        output_dir += '_' + args.exp_tag
    dirs = init_dir(output_dir, pathes=['configs', 'models', 'logs', 'output'])

    # copy all files to the results that have influence on it
    copy_tree("../highway-env", dirs['configs'])
    copy('configs/configs_sacd.json', dirs['configs'])
    copy(__file__, dirs['configs'])
    with open(dirs['configs']+ "args", "w") as f:
        for arg in sys.argv:
            f.write(f"{arg} ")

    # configure environment
    env = gym.make('merge-single-agent-v0')
    env.config.update(config['env_config'])

    # for curriculum learning start from difficulty 1
    curriculum_learning = config.get('curriculum', False)
    if curriculum_learning == True:
        env.config['traffic_density'] = 1

    # seed from commandline has priority over config
    seed_ = config.get('seed', 42) if args.seed == 0 else args.seed

    # configure callbacks
    eval_env = gym.make('merge-single-agent-v0')
    eval_env.config.update(config['env_config'])
    eval_env.config["traffic_density"] = 3
    #eval_callback = EvalCallback(eval_env, log_path=dirs['logs'], eval_freq=500, deterministic=True, render=False)
    custom_eval = CustomEvalCallback(dirs['logs'])

    eval_callback = EveryNTimesteps(n_steps=500, callback=custom_eval)


    checkpoint_log_speed = TensorboardCallback()
    model = SACD('MlpPolicy', env,
                  policy_kwargs=dict(net_arch=[256, 256]),
                  seed=seed_,
                  learning_rate=5e-4,
                  train_freq=4,
                  target_update_interval=8,
                  batch_size=256,
                  gamma=0.99,
                  verbose=1,
                  tensorboard_log=dirs['logs'],
                  device=f"cuda:{args.gpu}")

    callback_list = CallbackList([eval_callback, checkpoint_log_speed])

    # configure logging
    custom_logger = configure(dirs['logs'], ["stdout", "csv", "tensorboard"])
    model.set_logger(custom_logger)

    # split up total learning steps when using curriculum learning
    learn_steps = 10e5
    if curriculum_learning == True:
        learn_steps = 3e5

    model.learn(int(learn_steps), tb_log_name=args.exp_tag + f"_seed_{seed_}", callback=callback_list)

    if curriculum_learning == True:
        env.config['traffic_density'] = 2
        model.learn(int(3e5), tb_log_name=args.exp_tag + f"_seed_{seed_}", reset_num_timesteps=False, callback=checkpoint_log_speed)
        env.config['traffic_density'] = 3
        model.learn(int(4e5), tb_log_name=args.exp_tag + f"_seed_{seed_}", reset_num_timesteps=False, callback=checkpoint_log_speed)

    model.save(dirs['models'] + f"/model_{args.exp_tag}_seed_{seed_}")

if __name__ == "__main__":
    args = parse_args()
    train(args)

