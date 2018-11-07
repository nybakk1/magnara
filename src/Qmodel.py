import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt

episodes = 2000
timesteps = 500
average_size = 100


class Qmodel:
    def __init__(self, env, bucket=(1, 1, 6, 12)):
        self.env = env
        self.action_size = env.action_space.n
        self.observation_size = env.observation_space.shape[0]

        self.bucket = bucket
        self.Q = self.buildQ()

        self.discount_factor = 0.1      # Increases as episodes go on to weight later actions less.
        self.discount_fact_inc = 0.002  # How much discount factor increases.
        self.discount_fact_max = 1.0    # Maximum limit for discount factor.
        self.epsilon = 1.0              # Chance to explore.
        self.epsilon_min = 0.01         # Minimum chance to explore.
        self.epsilon_decay = 0.995      # Exploration decay factor.
        self.learning_rate = 0.1        # Learning rate.

    def bucketize(self, state):
        """
        Bucketize the state to be discrete.
        Credit: https://gist.github.com/n1try/af0b8476ae4106ec098fea1dfe57f578
        :param state: state from environment.
        :return: discrete ndarray
        """
        upper = [self.env.observation_space.high[0], 1.0, self.env.observation_space.high[2], math.radians(41.8)]
        lower = [self.env.observation_space.low[0], -1.0, self.env.observation_space.low[2], -math.radians(41.8)]
        ratio = [(state[i] + abs(lower[i])) / (upper[i] - lower[i]) for i in range(len(state))]
        new_state = [int(round((self.bucket[i] - 1) * ratio[i])) for i in range(len(state))]
        new_state = [min(self.bucket[i] - 1, max(0, new_state[i])) for i in range(len(state))]
        return np.asarray(tuple(new_state))

    def policy(self, state):
        """
        Policy funtion to figure out what action to take.
        :param state: a discrete
        :return: an action
        """
        # Return random action
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        return np.argmax(self.Q[self.hash(state)])

    def hash(self, state):
        """
        Hash a discrete state to a unique integer value.
        :param state: list of integer values.
        :return: unique integer hash.
        """
        hash = 0
        hash += self.bucket[1] * self.bucket[2] * self.bucket[3] * state[0]
        hash += self.bucket[2] * self.bucket[3] * state[1]
        hash += self.bucket[3] * state[2]
        hash += state[3]
        return hash

    def buildQ(self):
        """
        Build the initial Q-table, ready for indexing.
        :return: the new Q-table.
        """
        Q = []
        for i in range(self.bucket[0] * self.bucket[1] * self.bucket[2] * self.bucket[3]):
            Q.append([0, 0])
        return Q

    def updateQ(self, state, next_state, action, reward):
        """
        Updates Q-table
        :param state: list of integer values.
        :param next_state: list of integer values.
        :param action: integer value.
        :param reward: integer value.
        """
        state = self.hash(state)
        next_state = self.hash(next_state)
        self.Q[state][action] = (1 - self.learning_rate) * self.Q[state][action] + self.learning_rate * (reward + self.discount_factor * np.max(self.Q[next_state]))


    def run(self, episodes=500, timesteps=200, average_size=50):
        """
        Run model, OpenAI gym simulates an environment and the agent starts to learn.
        The rolling average of the scores is plotted at the end.
        :param episodes: positive integer
        :param timesteps: positive integer
        :param average_size: positive integer
        """
        scores = []             # Save scores to calculate average scores.
        rolling_average = []    # Save average score for plotting.
        for e in range(episodes):
            state = self.bucketize(self.env.reset())            # Make state discrete.
            for ts in range(timesteps):
                #if e % 500 == 0: self.env.render()
                action = self.policy(state)                     # Figure out what action to do.
                next_state, reward, done, _ = env.step(action)  # Do the action.
                reward = reward if not done else -timesteps/2   # Punish for losing.
                next_state = self.bucketize(next_state)         # Make next_state discrete.

                self.updateQ(state, next_state, action, reward)     # Update Q-table.

                state = next_state
                if done or ts >= timesteps -1:
                    scores.append(ts)
                    if e > average_size:
                        average = np.average(scores[e-average_size:e])
                        print(f'Episode {e + 1}/{episodes}, average: {average}')
                        rolling_average.append(average)
                    break
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.discount_factor = min(self.discount_factor + self.discount_fact_inc, self.discount_fact_max)

        # Plot rolling average
        x = [i + average_size for i in range(len(rolling_average))]
        plt.plot(x, rolling_average)
        plt.title(f"Rolling average of past {average_size} episodes.")
        plt.ylabel("Average score")
        plt.xlabel("Episodes")
        plt.ylim(0, timesteps)
        plt.show()


env = gym.make('CartPole-v1')
q_model = Qmodel(env)
q_model.run(episodes, timesteps, average_size)
