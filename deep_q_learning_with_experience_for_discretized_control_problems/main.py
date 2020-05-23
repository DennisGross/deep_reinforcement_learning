"""
Author: Dennis Gross
This agent can be used to solve discretized control problems from open ai gym.
This script is based on the research paper 'Playing Atari with Deep Reinforcement Learning'
Link: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
It is inspired by:
- https://github.com/openai/baselines/blob/master/baselines/deepq/deepq.py
- https://www.youtube.com/watch?v=5fHngyN8Qhw
"""

import gym
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


class ReplayBuffer():

    def __init__(self, buffer_size, input_dim):
        """
        Creates the experience replay buffer for the agent
        :param buffer_size: buffer size
        :param input_dim: state dimension
        """
        self.buffer_size = buffer_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.buffer_size, input_dim),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.buffer_size, input_dim),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.buffer_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.buffer_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.buffer_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        """
            Stores the transition into the replay buffer
            :param state: current state
            :param action: current action
            :param reward: current reward
            :param new_state: action taken in state leads to state new_state
            :param done: game over? true/false
            :return:
        """
        index = self.mem_cntr % self.buffer_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        :param batch_size: Batch size for training
        :return: training sample
        """
        max_mem = min(self.mem_cntr, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class Agent():
    def __init__(self, lr, gamma, number_of_actions, epsilon, batch_size,
                 input_dim, epsilon_dec=0.9996, epsilon_end=0.01,
                 buffer_size=1000000, model_file='model.h5', execution_mode = False):
        """

        :param lr: learning rate of neural network
        :param gamma: discount factor
        :param number_of_actions: number of actions
        :param epsilon: exploration factor
        :param batch_size: training batch size
        :param input_dim: input size
        :param epsilon_dec: epsilon decrease
        :param epsilon_end: lower bound of epsilon
        :param buffer_size: size of replay buffer
        :param model_file: save/load model from this file.
        :param execution_mode: training ( or execution
        """
        self.action_space = [i for i in range(number_of_actions)]
        self.gamma = gamma
        if execution_mode:
            self.epsilon = 0
            self.eps_dec = 0
            self.eps_min = 0
        else:
            self.epsilon = epsilon
            self.eps_dec = epsilon_dec
            self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = model_file
        self.memory = ReplayBuffer(buffer_size, input_dim)
        self.q_eval = self.build_dqn(lr, number_of_actions, input_dim)

    def build_dqn(self, lr, n_actions, input_dim):
        """
        Build the neural network that approximate the q-values
        :param lr: learning rate
        :param n_actions: output dimension of the neural network
        :param input_dim: input dimension of the neural network
        :return:
        """
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=input_dim),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(n_actions, activation=None)])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')

        return model

    def store_transition(self, state, action, reward, new_state, done):
        """
        Stores the transition into the replay buffer
        :param state: current state
        :param action: current action
        :param reward: current reward
        :param new_state: action taken in state leads to state new_state
        :param done: game over? true/false
        :return:
        """
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        """
        Choose an action.
        :param observation: current state
        :return:
        """
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)

        return action

    def learn(self):
        """
        Trains the neural network
        :return:
        """
        # Don't learn if replay buffer has got less number of elements than the batch size
        if self.memory.mem_cntr < self.batch_size:
            return

        # Get random transitions from experience replay buffer
        states, actions, rewards, states_, dones = self.memory.sample_buffer(self.batch_size)

        # q_target the same shape that q_eval
        q_target = np.copy(self.q_eval.predict(states))
        # Needed for accessing the batch
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        # Calculate best possible reward in the next state
        q_next = self.q_eval.predict(states_)
        # First create new Q-values
        q_target[batch_index, actions] = rewards + self.gamma * np.max(q_next, axis=1) * dones
        # Update neural network with new Q-values
        self.q_eval.train_on_batch(states, q_target)

        # Decrease exploration probability
        self.epsilon = self.epsilon * self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_model(self):
        """
        Save the neural network model
        :return:
        """
        self.q_eval.save(self.model_file)

    def load_model(self):
        """
        Load the neural network/brain into the agent
        :return:
        """
        try:
            self.q_eval = load_model(self.model_file)
        except:
            print("Could not load", self.model_file)


if __name__ == '__main__':
    environment_name = "MountainCar-v0" # or: Acrobot-v1, CartPole-v1, MountainCar-v0, and Pendulum-v0
    env = gym.make(environment_name)
    n_games = 10000
    agent = Agent(gamma=0.99, epsilon=1.0, lr=0.001, epsilon_dec=0.99997, epsilon_end=0.01,
                  input_dim=env.observation_space.shape[0],
                  number_of_actions=env.action_space.n, buffer_size=1000000, batch_size=64,
                  execution_mode = False, model_file=environment_name+".h5")
    agent.load_model()
    scores, eps_history = [], []
    last_ag_score = 0
    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            env.render()
            agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(score)
        if i % 10 == 0 and i > 0:
            agent.save_model()
        avg_score = np.mean(scores[-100:])
        print("episode", i, "score %.1f" % score, "average score %.1f" % avg_score, "epsilon %f" % agent.epsilon)
