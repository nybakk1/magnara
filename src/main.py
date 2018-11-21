from DQNmodel import *
from Qmodel import *
from Plot import plot

import gym

episodes = 1000
batch_size = 64

env = gym.make("CartPole-v1")
dqn = DeepQAgent(env, episodes, batch_size)
q_model = Qmodel(env)

timesteps = 500
average_size = 100


dqn_run = dqn.run(timesteps, batch_size, average_size, True, 1)
dqn_run2 = dqn.run(timesteps, batch_size, average_size, False, 2)
q_run = q_model.run(True, episodes, timesteps, average_size)
q_run2 = q_model.run(False, episodes, timesteps, average_size)

plot(data=[([i + average_size for i in range(len(dqn_run[1]))], dqn_run[1]),
           ([i + average_size for i in range(len(q_run[1]))], q_run[1])],
     legend=["DQN-Agent", "Q-learning"],
     yMax=timesteps,
     title=f"Rolling average of past {average_size} episodes.",
     labels=("Episode", "Average score"))

plot(data=[([i + average_size for i in range(len(dqn_run2[1]))], dqn_run2[1]),
           ([i + average_size for i in range(len(q_run2[1]))], q_run2[1])],
     legend=["DQN-Agent", "Q-learning"],
     yMax=timesteps,
     title=f"Rolling average of past {average_size} episodes.",
     labels=("Episode", "Average score"))
