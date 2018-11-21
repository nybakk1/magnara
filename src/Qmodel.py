import numpy as np
import math
import random
import time


class Qmodel:
    def __init__(self, env, bucket=(1, 1, 6, 12)):
        self.env = env
        self.action_size = env.action_space.n
        self.observation_size = env.observation_space.shape[0]

        self.bucket = bucket
        self.Q = self.build_q()

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

    def policy(self, state, explore=True):
        """
        Policy funtion to figure out what action to take.
        :param state: a discrete
        :return: an action
        """
        # Return random action
        if explore and np.random.rand() <= self.epsilon:
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

    def build_q(self):
        """
        Build the initial Q-table, ready for indexing.
        :return: the new Q-table.
        """
        Q = []
        for i in range(self.bucket[0] * self.bucket[1] * self.bucket[2] * self.bucket[3]):
            Q.append([0, 0])
        return Q

    def update_q(self, state, next_state, action, reward):
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

    def run(self, run_name, train=True, episodes=500, max_score=200, average_size=100):
        """
        Run model, OpenAI gym simulates an environment and the agent starts to learn.
        The rolling average of the scores is plotted at the end.
        :param run_name: String for naming the run
        :param train: boolean whether the model should do exploring and train.
        :param episodes: positive integer how many episodes the model should train on.
        :param max_score: positive integer how long the model will try to keep the pole up.
        :param average_size: positive integer how many previous episodes the rolling average should use.
        """
        scores = []             # Save scores to calculate average scores.
        episode_solved = -1
        found_solved = False
        start_time = int(time.time())
        for e in range(episodes):
            state = self.bucketize(self.env.reset())            # Make state discrete.
            for timestep in range(max_score):
                action = self.policy(state, train)                      # Figure out what action to do.
                next_state, reward, done, _ = self.env.step(action)     # Do the action.
                reward = reward if not done else (-max_score/2)         # Punish for losing.
                next_state = self.bucketize(next_state)                 # Make next_state discrete.

                self.update_q(state, next_state, action, reward) if train else None     # Update Q-table.

                state = next_state
                if done or timestep >= max_score - 1:
                    score = timestep + 1
                    scores.append(score)
                    if e > average_size:
                        average = np.average(scores[e-average_size:e])
                        print(f'Run: {run_name}\tEpisode {e + 1}/{episodes}\tscore: {score}\taverage: {average}')
                        if average >= max_score * 0.95 and not found_solved:
                            episode_solved = e - 99
                            found_solved = True
                    else:
                        print(f'Run: {run_name}\tEpisode {e + 1}/{episodes}\tscore: {score}')
                    break
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.discount_factor = min(self.discount_factor + self.discount_fact_inc, self.discount_fact_max)

        duration = int(time.time()) - start_time
        print(f"Problem solved after {episode_solved} episodes")
        return scores, duration
