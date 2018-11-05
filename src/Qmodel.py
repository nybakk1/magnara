import gym
import numpy as np
import math
import random
import matplotlib.pyplot as plt

episodes = 1000
timesteps = 200


class Qmodel:
    def __init__(self, env, bucket=(1, 1, 6, 12)):
        self.env = env
        self.action_size = env.action_space.n
        self.observation_size = env.observation_space.shape[0]

        self.bucket = bucket
        self.Q = self.buildQ()   # Q[state][action] = (1-learning_rate) * Q[state][action] + learning_rate * (reward + discount_factor * max(Q[nextState])
                                  # Q[state][action] += learning_rate * (-Q[state][action] + reward + discount_factor + max(Q[nextState]))

        self.discount_factor = 0.1     # Increases as episodes go on to weight later actions less.
        self.epsilon = 1.0              # Chance to explore.
        self.epsilon_min = 0.01         # Minimum chance to explore.
        self.epsilon_decay = 0.995       # Exploration decay factor.
        self.learning_rate = 0.1      # Learning rate.

    def bucketize(self, state):
        """
        Bucketize the state to be discrete.
        Credit: TODO: implement credit here
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

        :param state:
        :return:
        """
        # Return random action
        if np.random.rand() <= self.epsilon:
            # print("RANDOM")
            return random.randrange(self.action_size)

        return np.argmax(self.Q[self.hash(state)])

        #if self.Q[hash(state)] is None:




    def hash(self, state):
        """

        :param state:
        :return:
        """
        hash = 0
        hash += self.bucket[1] * self.bucket[2] * self.bucket[3] * state[0]
        hash += self.bucket[2] * self.bucket[3] * state[1]
        hash += self.bucket[3] * state[2]
        hash += state[3]
        return hash

    def buildQ(self):
        """

        :return:
        """
        Q = []
        for i in range(self.bucket[0] * self.bucket[1] * self.bucket[2] * self.bucket[3]):
            Q.append([0, 0])
        return Q

    def run(self, episodes=500, timesteps=200, average_size=50):
        """

        :param episodes:
        :param timesteps:
        :return:
        """
        scores = []
        rolling_average = []
        for e in range(episodes):
            state = self.bucketize(self.env.reset())
            for ts in range(timesteps):
                action = self.policy(state)                     # Figure out what action to do.
                next_state, reward, done, _ = env.step(action)  # Do action
                reward = reward if not done else -10
                next_state = self.bucketize(next_state)
                self.Q[self.hash(state)][action] = (1 - self.learning_rate) * self.Q[self.hash(state)][action] + self.learning_rate * (reward + self.discount_factor * np.argmax(self.Q[self.hash(next_state)]))
                # self.Q[self.hash(state)][action] += self.learning_rate * (-self.Q[self.hash(state)][action] + reward + self.discount_factor + np.argmax(self.Q[self.hash(next_state)])) # Lernningnne ehth state s is bad things

                state = next_state
                if done or ts >= timesteps -1:
                    scores.append(ts)
                    #print(f'Episode {e}: Score: {ts}')
                    if e > average_size:
                        average = np.average(scores[e-average_size:e])
                        print(f'Episode {e+1}/{episodes}, average: {average}')
                        rolling_average.append(average)
                    break
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            if self.discount_factor*1.01 < 1:
                self.discount_factor *= 1.01
            print(self.discount_factor)

        # Plot rolling average
        x = [i + average_size for i in range(len(rolling_average))]
        plt.plot(x, rolling_average)
        plt.title("Rolling average of past " + str(average_size) + " episodes.")
        plt.ylabel("Average score")
        plt.xlabel("Episodes")
        plt.ylim(0, timesteps)
        plt.show()


env = gym.make('CartPole-v1')
q_model = Qmodel(env)
q_model.run(episodes, timesteps)
