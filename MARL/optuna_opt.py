import gymnasium as gym
import sys
import warnings
import argparse
import json
from distutils.dir_util import copy_tree
from shutil import copy
from datetime import datetime
from common.utils import init_dir
from functools import partial

warnings.filterwarnings('ignore')

sys.path.append("../highway-env")
import highway_env
from sb3_contrib import SACD
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

from optuna.integration.tensorboard import TensorBoardCallback

import torch

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

N_TRIALS = 2
N_STARTUP_TRIALS = 5
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3

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

def sample_lidar_params(trial: optuna.Trial):
    """Sampler for lidar perparameters."""
    cells = trial.suggest_int("cells", 10, 32)

    # Display true values.
    trial.set_user_attr("cells_", cells)

    return {
        "cells": cells,
    }

class TrialEvalCallback(EvalCallback):
    """Callback used for evaluating and reporting a trial."""

    def __init__(
        self,
        eval_env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need.
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True

def objective(trial, args):
    base_dir = args.base_dir
    config_dir = args.config_dir

    # load json config
    with open(config_dir) as f:
        config = json.load(f)

    # update with the optuna suggestions for run
    config['env_config']['observation'].update(sample_lidar_params(trial))
    print(config)

    # create an experiment folder
    now = datetime.utcnow().strftime("%b_%d_%H_%M_%S")
    output_dir = base_dir + now
    if args.exp_tag != '':
        output_dir += '_' + args.exp_tag
        
    output_dir += '_' + str(trial.number)
    dirs = init_dir(output_dir, pathes=['configs', 'models', 'logs', 'output'])

    # copy all files to the results that have influence on it
    print(dirs['configs'])
    copy_tree("../highway-env", dirs['configs'])
    copy('configs/configs_sacd.json', dirs['configs'])
    copy(__file__, dirs['configs'])
    with open(dirs['configs']+ "args", "w") as f:
        for arg in sys.argv:
            f.write(f"{arg} ")

    # configure environment
    env = gym.make('merge-single-agent-v0', config=config['env_config'])
    env.config.update(config['env_config'])

    # for curriculum learning start from difficulty 1
    curriculum_learning = config.get('curriculum', False)
    if curriculum_learning == True:
        env.config['traffic_density'] = 1

    # seed from commandline has priority over config
    seed_ = config.get('seed', 42) if args.seed == 0 else args.seed

    eval_env = gym.make('merge-single-agent-v0', config=config['env_config'])
    eval_env.config.update(config['env_config'])
    eval_env.config["traffic_density"] = 3

    # Create the callback that will periodically evaluate and report the performance.
    eval_callback = TrialEvalCallback(
        eval_env, trial, n_eval_episodes=N_EVAL_EPISODES, eval_freq=EVAL_FREQ, deterministic=True
    )

    model = SACD('MultiInputPolicy', env,
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

    # configure logging
    custom_logger = configure(dirs['logs'], ["stdout", "csv", "tensorboard"])
    model.set_logger(custom_logger)

    # split up total learning steps when using curriculum learning
    learn_steps = 10e5
    if curriculum_learning == True:
        learn_steps = 3e5

    model.learn(int(learn_steps), tb_log_name=args.exp_tag + f"_seed_{seed_}", log_interval=20)

    if curriculum_learning == True:
        env.config['traffic_density'] = 2
        model.learn(int(3e5), tb_log_name=args.exp_tag + f"_seed_{seed_}", reset_num_timesteps=False, callback=callback_list, log_interval=20)
        env.config['traffic_density'] = 3
        model.learn(int(4e5), tb_log_name=args.exp_tag + f"_seed_{seed_}", reset_num_timesteps=False, callback=callback_list, log_interval=20)

    model.save(dirs['models'] + f"/model_{args.exp_tag}_seed_{seed_}")

    # Free memory
    env.close()
    eval_env.close()

    return eval_callback.last_mean_reward

if __name__ == "__main__":
    args = parse_args()

    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used.
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3)

    tensorboard_callback = TensorBoardCallback("optuna_logs/", metric_name="last_mean_reward")

    #study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    study = optuna.create_study(study_name=args.exp_tag, sampler=sampler, pruner=pruner, direction="maximize", storage="mysql+pymysql://optuna@localhost/optuna", load_if_exists=True)
    try:
        objective = partial(objective, args=args)
        study.optimize(objective, n_trials=N_TRIALS, timeout=600, callbacks=[tensorboard_callback])
        #study.optimize(objective, n_trials=N_TRIALS, timeout=600, n_jobs=8)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))
