from DQNmodel import *
from Qmodel import *
from Plot import plot

import gym

dqn = DeepQAgent(gym.make("CartPole-v1"))
q_model = Qmodel(gym.make("CartPole-v1"))

episodes = 1000
timesteps = 500
batch_size = 32
average_size = 100
explore = True

dqn_run = dqn.run(episodes, timesteps, batch_size, average_size, explore)
q_run = q_model.run(explore, episodes, timesteps, average_size)

plot(data=[([i + average_size for i in range(len(dqn_run[1]))], dqn_run[1]),
           ([i + average_size for i in range(len(q_run[1]))], q_run[1])],
     legend=["DQN-Agent", "Q-learning"],
     title=f"Rolling average of past {average_size} episodes.",
     labels=("Episode", "Average score"))
