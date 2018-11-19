from DQNmodel import *
from Qmodel import *
from Plot import plot

import gym

episodes = 2000

dqn = DeepQAgent(gym.make("CartPole-v1"), episodes)
q_model = Qmodel(gym.make("CartPole-v1"))

timesteps = 500
batch_size = 32
average_size = 100
explore = True

dqn_run = dqn.run(timesteps, batch_size, average_size, explore, 1)
q_run = q_model.run(explore, episodes, timesteps, average_size)

plot(data=[([i + average_size for i in range(len(dqn_run[1]))], dqn_run[1]),
           ([i + average_size for i in range(len(q_run[1]))], q_run[1])],
     legend=["DQN-Agent", "Q-learning"],
     title=f"Rolling average of past {average_size} episodes.",
     labels=("Episode", "Average score"))

explore = False
dqn_run2 = dqn.run(timesteps, batch_size, average_size, explore, 2)
q_run2 = q_model.run(explore, episodes, timesteps, average_size)

plot(data=[([i + average_size for i in range(len(dqn_run2[1]))], dqn_run2[1]),
           ([i + average_size for i in range(len(q_run2[1]))], q_run2[1])],
     legend=["DQN-Agent", "Q-learning"],
     title=f"Rolling average of past {average_size} episodes.",
     labels=("Episode", "Average score"))

