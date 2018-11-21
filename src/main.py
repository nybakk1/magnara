from DQNmodel import *
from Qmodel import *
from Plot import plot

import gym

episodes = 2000
batch_size = 64
max_score = 500
average_size = 100

env = gym.make("CartPole-v1")
dqn = DeepQAgent(env, episodes, batch_size)
q_model = Qmodel(env)

dqn_run = dqn.run("DQN-Train", max_score, average_size, True)
dqn_run2 = dqn.run("DQN-Test", max_score, average_size, False)
q_run = q_model.run("Q-Train", True, episodes, max_score, average_size)
q_run2 = q_model.run("Q-Test", False, episodes, max_score, average_size)

print(f"Duration: {dqn_run[1]} seconds\n"
      f"Scores: {dqn_run[0]}")
print(f"Duration: {dqn_run2[1]} seconds\n"
      f"Scores: {dqn_run2[0]}")
print(f"Duration: {q_run[1]} seconds\n"
      f"Scores: {q_run[0]}")
print(f"Duration: {q_run2[1]} seconds\n"
      f"Scores: {q_run2[0]}")

# plot(data=[([i + average_size for i in range(len(dqn_run[1]))], dqn_run[1]),
#            ([i + average_size for i in range(len(q_run[1]))], q_run[1])],
#      legend=["DQN-Agent", "Q-learning"],
#      yMax=timesteps,
#      title=f"Rolling average of past {average_size} episodes.",
#      labels=("Episode", "Average score"))
#
# plot(data=[([i + average_size for i in range(len(dqn_run2[1]))], dqn_run2[1]),
#            ([i + average_size for i in range(len(q_run2[1]))], q_run2[1])],
#      legend=["DQN-Agent", "Q-learning"],
#      yMax=timesteps,
#      title=f"Rolling average of past {average_size} episodes.",
#      labels=("Episode", "Average score"))
