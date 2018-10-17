from keras.models import Sequential
from keras.layers import Dense, activations
from keras.optimizers import Adam
from collections import deque

import gym as G
import numpy as np
import random
import matplotlib.pyplot as plt

bat_size = 32
episodes = 500

class DeepQAgent():
    def __init__(self, action_space, observation_space):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01  # ?
        self.epsilon_decay = 0.995  # ?
        self.learning_rate = 1e-3
        self.action_space = action_space
        self.observation_space = observation_space
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(32, input_dim=self.observation_space, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_space, activation='sigmoid'))
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

    def never_forget(self, state, action, reward, done, next_state):
        self.memory.append((state, action, reward, done, next_state))

    def just_do_it(self, state):
        # TODO: Random Action Probability (Read: RAP)
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # This will fuck up, maybe.

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, done, next_state in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state,target_f,epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


env = G.make('CartPole-v1')
agent = DeepQAgent(env.action_space.n, env.observation_space.shape[0])
scores = []
for e in range(episodes):
    state = env.reset()

    state = np.reshape(state, [1, env.observation_space.shape[0]])
    for ts in range(500):
        env.render()
        action = agent.just_do_it(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, env.observation_space.shape[0]])
        state = next_state
        if done:
            scores.append(ts)
            agent.never_forget(state, action, ts, done, next_state)
            print('Episode {}/{}, score: {}'.format(e, episodes, ts))
            break
    agent.replay(bat_size)

x = [i for i in range(0, episodes)]
plt.plot(scores)
plt.show()
