import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("../highway-env")
import highway_env

from Discrete_SAC_Agent import SACAgent

TRAINING_EVALUATION_RATIO = 50
RUNS = 1
EPISODES_PER_RUN = 5000
MAX_STEPS_PER_EPISODE = 200

if __name__ == "__main__":
    # env = gym.make("CartPole-v1")
    env = gym.make('merge-single-agent-v0')
    env.config["safety_guarantee"] = False
    agent_results = []
    for run in range(RUNS):
        agent = SACAgent(env)
        # agent.load('models/', 2, train_mode=True)
        run_results = []
        for episode_number in tqdm(range(EPISODES_PER_RUN), smoothing=0.7):
            # print('\r', f'Run: {run + 1}/{RUNS} | Episode: {episode_number + 1}/{EPISODES_PER_RUN}', end=' ')
            evaluation_episode = episode_number % TRAINING_EVALUATION_RATIO == 0
            episode_reward = 0
            state, _ = env.reset()
            state = state.flatten()
            done = False
            i = 0
            while not done and i < MAX_STEPS_PER_EPISODE:
                i += 1
                action = agent.get_next_action(state, evaluation_episode=evaluation_episode)
                # print(f"{action=}")
                next_state, reward, done, _, info = env.step(action)
                next_state = next_state.flatten()
                if not evaluation_episode:
                    agent.train_on_transition(state, action, next_state, reward, done)
                else:
                    episode_reward += reward
                state = next_state
            if evaluation_episode:
                run_results.append(episode_reward)
        agent_results.append(run_results)
        agent.save('models/', run)

    env.close()

    n_results = EPISODES_PER_RUN // TRAINING_EVALUATION_RATIO
    results_mean = [np.mean([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    results_std = [np.std([agent_result[n] for agent_result in agent_results]) for n in range(n_results)]
    mean_plus_std = [m + s for m, s in zip(results_mean, results_std)]
    mean_minus_std = [m - s for m, s in zip(results_mean, results_std)]

    x_vals = list(range(len(results_mean)))
    x_vals = [x_val * (TRAINING_EVALUATION_RATIO - 1) for x_val in x_vals]

    ax = plt.gca()
    ax.set_ylim([-150, 100])
    ax.set_ylabel('Episode Score')
    ax.set_xlabel('Training Episode')
    ax.plot(x_vals, results_mean, label='Average Result', color='blue')
    ax.plot(x_vals, mean_plus_std, color='blue', alpha=0.1)
    ax.plot(x_vals, mean_minus_std, color='blue', alpha=0.1)
    ax.fill_between(x_vals, y1=mean_minus_std, y2=mean_plus_std, alpha=0.1, color='blue')
    plt.legend(loc='best')
    plt.show()
    plt.savefig("qianqian_params_no_grad_clipping.png")
