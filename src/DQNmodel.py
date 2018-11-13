from keras.models import Sequential
from keras.layers import Dense, activations
from keras.optimizers import Adam
from collections import deque
from sklearn import preprocessing

import gym as G
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import importlib

# Force TensorFlow to run on CPU. Comment out these lines
# if you don't use TensorFlow-GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

episodes = 2000
max_score = 500
batch_size = 32
average_size = 100
render = False
when_should_the_code_render_the_cart_pole_v1 = 500

# Increase batch-size
inc_every_episode = 50
batch_size_inc = 10


class DeepQAgent:
    def __init__(self, env):
        # hyperparameters
        self.memory = deque(maxlen=2000)
        self.env = env

        self.discount_factor = 0.1      # Increases as episodes go on to weight later actions less.
        self.discount_fact_inc = 0.002  # How much discount factor increases.
        self.discount_fact_max = 1.0    # Maximum limit for discount factor.
        self.epsilon = 1.0              # Chance to explore.
        self.epsilon_min = 0.01         # Minimum chance to explore.
        self.epsilon_decay = 0.995      # Exploration decay factor.
        self.learning_rate = 1e-2       # Learning rate.
        self.learning_rate_decay = 0.000001  # Learning rate decay

        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape[0]
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.observation_space, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        # TODO: Find the perfect decay value
        model.compile(optimizer=Adam(lr=self.learning_rate, decay=self.learning_rate_decay), loss='mse')
        return model

    # def normalize(self, state):
    #     high = [4.8, 3.4028235e+3, 4.1887903e-01, 3.4028235e+3]
    #     low = [-4.8, -3.4028235e+3, -4.1887903e-01, -3.4028235e+3]
    #     # getcontext().prec = 6
    #     for i in range(len(state)):
    #         value = state[i]
    #         min = low[i]
    #         max = high[i]
    #         state[i] = ((value - min) / (max - min))
    #     return state

    # def normalize(self, state):
    #     for i in range(len(state)):
    #         foo = preprocessing.minmax_scale(state, (0, 1))
    #
    #     # foobar = foo.fit_transform(state)
    #     return foo

    # lagrer staten i minnet
    def save_state(self, state, action, reward, done, next_state):
        """
        Save an experience in memory for use later when training

        :param state:
        :param action:
        :param reward:
        :param done:
        :param next_state:
        """
        self.memory.append((state, action, reward, done, next_state))

    def policy(self, state, explore=True):
        """
        Policy funtion to figure out what action to take.

        :param state:
        :return: an action
        """
        # TODO: Random Action Probability (Read: RAP)
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # This will fuck up, maybe.

    def train(self, batch_size, e):
        """
        Trains the model by picking a random batch of earlier experiences from memory

        :param batch_size: positive integer
        """
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)
        if e % inc_every_episode is 0:
            batch_size += batch_size_inc

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, done, next_state in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.discount_factor = min(self.discount_factor + self.discount_fact_inc, self.discount_fact_max)

    def run(self, episodes=1000, timesteps=500, batch_size=32, average_size=100, explore=True):
        """
        Run model, OpenAI gym simulates an environment and the agent starts to learn.
        The rolling average of the scores is plotted at the end.

        :param episodes: positive integer
        :param timesteps: positive integer
        :param batch_size: positive integer
        :param average_size: positive integer
        """
        scores = []             # Save scores to calculate average scores.
        rolling_average = []    # Save average score for plotting.
        for e in range(episodes):
            state = self.env.reset()
            # state = self.normalize(state)
            state = np.reshape(state, [1, self.observation_space])
            for ts in range(timesteps):
                if e % when_should_the_code_render_the_cart_pole_v1 is 0 and render is True:
                    self.env.render()
                action = self.policy(state, explore)
                next_state, reward, done, _ = self.env.step(action)     # Do the action
                reward = reward if not done else -timesteps             # Punish for losing.
                # next_state = self.normalize(next_state)
                next_state = np.reshape(next_state, [1, self.observation_space])
                self.save_state(state, action, reward, done, next_state)

                state = next_state
                if done or ts >= timesteps - 1:
                    scores.append(ts)
                    if e > average_size:
                        average = np.average(scores[e-average_size:e])
                        print(f'Episode {e + 1}/{episodes}, average: {average}')
                        rolling_average.append(average)
                    break
            self.train(batch_size,e)

        return scores, rolling_average


env = G.make('CartPole-v1')
agent = DeepQAgent(env)
train = agent.run(episodes, max_score, batch_size, average_size, True)
test = agent.run(episodes, max_score, batch_size, average_size, False)

from Plot import plot
plot(data=[[[i+average_size for i in range(len(train[1]))], train[1]], [[i+average_size for i in range(len(test[1]))], test[1]]],
     legend=["Training run", "Test run"],
     title=f"Rolling average of past {average_size} episodes.",
     labels=["Episode", "Average score"])
