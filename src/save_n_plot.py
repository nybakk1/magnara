from src.DQNmodel import *
from src.Qmodel import *

import gym
import json
import numpy as np
import matplotlib.pyplot as plt


def save_json(file_name, method, runs):
    env = gym.make("CartPole-v1")

    train_data = {}
    test_data = {}
    if method is "dqn":
        for i in range(runs):
            dqn = DeepQAgent(env)
            train_scores, train_duration = dqn.run(i)
            dqn.episodes = 200
            test_scores, test_duration = dqn.run(i, train=False)
            train_data[f'Run {i}'] = {
                'scores': train_scores,
                'duration': train_duration
            }
            test_data[f'Run {i}'] = {
                'scores': test_scores,
                'duration': test_duration
            }
    else:
        for i in range(runs):
            q_model = Qmodel(env)
            train_scores, train_duration = q_model.run(i)
            q_model.episodes = 200
            test_scores, test_duration = q_model.run(i, train=False)
            train_data[f'Run {i}'] = {
                'scores': train_scores,
                'duration': train_duration
            }
            test_data[f'Run {i}'] = {
                'scores': test_scores,
                'duration': test_duration
            }

    with open(f'train-{file_name}' if file_name.endswith(".json") else f'train-{file_name}.json', 'w') as outfile:
        json.dump(train_data, outfile)
    with open(f'test-{file_name}' if file_name.endswith(".json") else f'test-{file_name}.json', 'w') as outfile:
        json.dump(test_data, outfile)


def plot_json(file_name, title="Plot"):
    with open(file_name if file_name.endswith(".json") else f'{file_name}.json') as json_file:
        data = json.load(json_file)

    avg_duration = 0
    vector = 0
    for p in data:
        vector += np.array(data[p]['scores'])
        avg_duration += np.array(data[p]['duration'])

    vector = vector / len(data)
    # avg_duration = avg_duration / len(data)
    X = np.linspace(0, len(data['Run 0']['scores']), len(data['Run 0']['scores']), dtype=int)

    p1 = np.polyfit(X, vector, 4)

    plt.plot(vector, 'g', alpha=0.2)
    plt.plot(X, np.polyval(p1, X), 'r', label='Average score/episode')
    for p in data:
        plt.scatter(X, data[p]['scores'], s=2, alpha=0.2, c='#6cb4e6')
    plt.title(f'{title} ({len(data)} runs)')
    plt.legend()
    plt.ylabel('Score')
    plt.xlim(0)
    plt.ylim(0)
    plt.xlabel('Episode')
    plt.show()


save_json("dqn", "dqn", 25)
save_json("q", "q", 5)
plot_json('train-q', 'Q-learning')
plot_json('test-q', 'Q-learning')
plot_json('train-dqn', 'Deep Q Network')
plot_json('test-dqn', 'Deep Q Network')
