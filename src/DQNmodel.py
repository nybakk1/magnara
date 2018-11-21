from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from collections import deque

import numpy as np
import random
import time

# Force TensorFlow to run on CPU. Comment out these lines
# if you don't use TensorFlow-GPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

render = False      # TODO: Integrate into run
when_should_the_code_render_the_cart_pole_v1 = 200  # TODO: Integrate into run


class DeepQAgent:
    def __init__(self, env, episodes=1000, batch_size=64):
        # hyperparameters
        self.memory = deque(maxlen=2000)
        self.env = env
        self.batch_size = batch_size

        self.episodes = episodes
        self.discount_factor = 0.8      # Increases as episodes go on to weight later actions less.
        self.epsilon = 1.0              # Chance to explore.
        self.epsilon_min = 0.01         # Minimum chance to explore.
        self.epsilon_decay = np.e ** (np.log(self.epsilon_min / self.epsilon) / (episodes * 0.8)) # Exploration decay factor.
        self.learning_rate = 1e-3       # Learning rate.

        self.action_space = self.env.action_space.n
        self.observation_space = self.env.observation_space.shape[0]
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.observation_space))
        model.add(Dense(16, activation="relu"))
        model.add(Dense(self.action_space, activation='linear'))
        # TODO: Find the perfect decay value
        model.compile(optimizer=Adam(lr=self.learning_rate), loss='mse')
        return model

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
        Policy function to figure out what action to take.

        :param state:
        :return: an action
        """
        # TODO: Random Action Probability (Read: RAP)
        if explore and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)

        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def train(self):
        """
        Trains the model by picking a random batch of earlier experiences from memory
        """
        if len(self.memory) < self.batch_size:
            minibatch = random.sample(self.memory, len(self.memory))
        else:
            minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, done, next_state in minibatch:
            target = reward
            if not done:
                target = reward + self.discount_factor * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def run(self, run_name, timesteps=500, average_size=100, train=True):
        """
        Run model, OpenAI gym simulates an environment and the agent starts to learn.
        The rolling average of the scores is plotted at the end.

        :param run_name: String for naming the run
        :param timesteps: positive integer the time to keep the pole upright
        :param average_size: positive integer the amount of previous episodes
        :param train: boolean wheter the model should explore and train
        """
        scores = []             # Save scores to calculate average scores.
        rolling_average = []    # Save average score for plotting.
        episode_solved = -1
        found_solved = False
        start_time = int(time.time())   # Get current time to return duration of the run.
        for e in range(self.episodes):
            state = self.env.reset()
            # state = self.normalize(state)
            state = np.reshape(state, [1, self.observation_space])
            for ts in range(timesteps):
                if e % when_should_the_code_render_the_cart_pole_v1 is 0 and render is True:
                    self.env.render()
                action = self.policy(state, train)
                next_state, reward, done, _ = self.env.step(action)     # Do the action
                reward = reward if not done else -100             # Punish for losing.
                # next_state = self.normalize(next_state)
                next_state = np.reshape(next_state, [1, self.observation_space])
                self.save_state(state, action, reward, done, next_state) if train else None

                state = next_state
                if done or ts >= timesteps - 1:
                    scores.append(ts)
                    if e > average_size:
                        average = np.average(scores[e - average_size:e])
                        print(f'Run: {run_name}\tEpisode {e + 1}/{self.episodes}\tscore: {ts}\taverage: {average}')
                        rolling_average.append(average)
                        if average == float(timesteps - 1) and not found_solved:
                            episode_solved = e - average_size - 1
                            found_solved = True
                    else:
                        print(f'Run: {run_name}\tEpisode {e + 1}/{self.episodes}\tscore: {ts}')
                    break
            self.train() if train else None

        print(f"Problem solved after {episode_solved} episodes")
        duration = int(time.time()) - start_time
        return scores, duration
