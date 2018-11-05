# Force TensorFlow to run on CPU. Comment out these lines
# if you don't use TensorFlow-GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from keras.models import Sequential
from keras.layers import Dense, activations
from keras.optimizers import Adam
from collections import deque

import gym as G
import numpy as np
import math
import random
import matplotlib.pyplot as plt


class DeepQAgent():
    def __init__(self, env, buckets=(10, 12, 10, 12)):
        # hyperparameters
        self.memory = deque(maxlen=2000)
        self.buckets = buckets
        self.env = env
        self.gamma = 0.99  # was .95
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
        upper = []

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
        for e in range(episodes):
            state = env.reset()

            state = np.reshape(state, [1, self.observation_space])
            for ts in range(timesteps):
                # env.render()
                action = agent.perform_action(state)
                next_state, reward, done, _ = self.env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, self.observation_space])
                agent.save_state(state, action, reward, done, next_state)
                state = next_state
                if done or ts >= timesteps:
                    scores.append(ts)
                    print(f'Episode {e+1}/{episodes}, score: {ts}')
                    break
            agent.train(bat_size)

        x = [i for i in range(episodes)]
        z = np.polyfit(x, scores, 8)
        f = np.poly1d(z)
        y_new = f(x)
        plt.plot(x, scores, 'o', x, y_new)
        plt.show()


env = G.make('CartPole-v1')
agent = DeepQAgent(env)
agent.run()
