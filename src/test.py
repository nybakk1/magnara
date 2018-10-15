 # examples/quickstart.py

import numpy as np
import threading

from tensorforce.agents import PPOAgent
from tensorforce.agents.learning_agent import LearningAgent
from tensorforce.execution import Runner
from tensorforce.contrib.openai_gym import OpenAIGym

from pynput import keyboard

# Create an OpenAIgym environment
#CartPole-v0
env = OpenAIGym('BipedalWalker-v2', visualize=False)


def key_check():
    def toggle_vis():
        if env.visualize:
            env.visualize = False
        else:
            env.visualize = True

    def on_key_release(key):
        if key.char == 'v':
            toggle_vis()

    with keyboard.Listener(on_release=on_key_release) as listener:
        listener.join()


t1 = threading.Thread(target=key_check)
t1.start()

# Network as list of layers
network_spec = [
    dict(type='dense', size=32, activation='tanh'),
    dict(type='dense', size=32, activation='tanh')
]

batch_size = 4096

agent = PPOAgent(
    states=env.states,
    actions=env.actions,
    network=network_spec,
    # PPOAgent
    step_optimizer=dict(
        type='adam',
        learning_rate=1e-3
    ),
    optimization_steps=10,
    # Model
    scope='ppo',
    discount=0.99,
    # DistributionModel
    entropy_regularization=0.01,
    # PGModel
    baseline_mode=None,
    baseline=None,
    baseline_optimizer=None,
    gae_lambda=None,
    # PGLRModel
    likelihood_ratio_clipping=0.2
)

# agent = LearningAgent(
#     states=env.states,
#     actions=env.actions,
#     network=network_spec,
#     update_mode={'unit': "timesteps", 'batch_size': 64, 'frequency': 4},
#     memory={'type': 'replay', 'include_next_states': True, 'capacity': 1000*batch_size},
#     optimizer={'type': 'adam', 'learning_rate': 1e-3}
# )

# Create the runner
runner = Runner(agent=agent, environment=env)


# Callback function printing episode statistics
def episode_finished(r):
    print("Finished episode {ep} after {ts} timesteps (reward: {reward})".format(ep=r.episode, ts=r.episode_timestep,
                                                                                 reward=r.episode_rewards[-1]))
    return True


# Start learning
runner.run(episodes=300000, max_episode_timesteps=200, episode_finished=episode_finished)
runner.close()

# Print statistics
print("Learning finished. Total episodes: {ep}. Average reward of last 100 episodes: {ar}.".format(
    ep=runner.episode,
    ar=np.mean(runner.episode_rewards[-100:]))
)

