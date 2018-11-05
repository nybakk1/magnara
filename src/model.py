from keras.models import Sequential
from keras.layers import Dense, activations
from keras.optimizers import Adam
from collections import deque

import gym as G
import numpy as np
import math
import random
import matplotlib.pyplot as plt

# Force TensorFlow to run on CPU. Comment out these lines
# if you don't use TensorFlow-GPU
# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class DeepQAgent:
    def __init__(self, env, buckets=(6, 12, 6, 12)):
        # hyperparameters
        self.memory = deque(maxlen=2000)
        self.buckets = buckets
        self.env = env
        self.gamma = .99  # was .95
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # ?
        self.epsilon_decay = 0.995  # ?
        self.learning_rate = 1e-3
        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape[0]
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.observation_space, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def discretize(self, state):
        upper = [self.env.observation_space.high[0], 1.0, self.env.observation_space.high[2], math.radians(41.8)]
        lower = [self.env.observation_space.low[0], -1.0, self.env.observation_space.low[2], -math.radians(41.8)]
        ratio = [(state[i] + abs(lower[i])) / (upper[i] - lower[i]) for i in range(len(state))]
        new_state = [int(round((self.buckets[i] - 1) * ratio[i])) for i in range(len(state))]
        new_state = [min(self.buckets[i] - 1, max(0, new_state[i])) for i in range(len(state))]
        return np.asarray(tuple(new_state))

    # lagrer staten i minnet
    def save_state(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))

    # bestemmer og utfører den action som velges.
    def perform_action(self, state):
        # TODO: Random Action Probability (Read: RAP)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # This will fuck up, maybe.

    # trener opp modellen. Plukker ut en batch med states fra minnet.
    def train(self, batch_size):
        if len(self.memory) < batch_size:
            batch_size = len(self.memory)

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, done, next_state in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self, episodes=500, timesteps=200, bat_size=32):
        scores = []
        average_size = 50
        rolling_average = []
        for e in range(episodes):
            state = self.discretize(self.env.reset())

            state = np.reshape(state, [1, self.observation_space])
            for ts in range(timesteps):
                # env.render()
                action = agent.perform_action(state)
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.discretize(next_state)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, self.observation_space])
                # print(state, next_state)
                agent.save_state(state, action, reward, done, next_state)
                state = next_state
                if done or ts >= timesteps - 1:
                    scores.append(ts)
                    if e > average_size:
                        average = np.average(scores[e-average_size:e])
                        print(f'Episode {e+1}/{episodes}, average: {average}')
                        rolling_average.append(average)
                    # print(f'Episode {e+1}/{episodes}, score: {ts}')
                    break
            agent.train(bat_size)

        # Plot rolling average
        x = [i+average_size for i in range(len(rolling_average))]
        plt.plot(x, rolling_average, 'o')
        plt.title("Rolling average of 50 episodes.")
        plt.ylabel("Average score")
        plt.xlabel("Episodes")
        plt.show()


env = G.make('CartPole-v1')
agent = DeepQAgent(env)
agent.run()
