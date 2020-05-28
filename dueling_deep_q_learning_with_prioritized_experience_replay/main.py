from collections import deque
import torch
import gym
import random

class PrioritizedExperienceReplay:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.state_memory = deque(maxlen=self.memory_size)
        self.action_memory = deque(maxlen=self.memory_size)
        self.reward_memory = deque(maxlen=self.memory_size)
        self.new_state_memory = deque(maxlen=self.memory_size)
        self.terminal_memory = deque(maxlen=self.memory_size)
        self.priorities = deque(maxlen=self.memory_size)

    def add(self, state, action, reward, new_state, terminal):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.new_state_memory.append(new_state)
        self.terminal_memory.append(1-terminal)
        self.priorities.append(max(self.priorities,default=1))


    def contains_more_elements_than_batch_size(self, batch_size):
        return len(self.state_memory) > batch_size

    def get_probabilities(self, priority_scale):
        scaled_priorities = torch.tensor(self.priorities)**priority_scale + 0.001
        sample_probabilities = scaled_priorities / torch.sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1/len(self.state_memory)* 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def sample_experiences(self, batch_size, priority_scale=1.0):
        """
        random_indizes = torch.randperm(len(self.state_memory))[
                         :batch_size].tolist()  # torch.randint(0, len(self.memory), (batch_size,)).tolist()
        """
        sample_size = batch_size
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.priorities)), k=sample_size, weights=sample_probs)
        importance = self.get_importance(sample_probs[sample_indices])
        return [self.state_memory[i] for i in sample_indices], \
               [self.action_memory[i] for i in sample_indices], \
               [self.reward_memory[i] for i in sample_indices], \
               [self.new_state_memory[i] for i in sample_indices], \
               [self.terminal_memory[i] for i in sample_indices], \
               importance


class DuelingDeepQNetwork(torch.nn.Module):

    def __init__(self, input_dimensions, action_dimensions, learning_rate):
        super(DuelingDeepQNetwork, self).__init__()
        self.input_dimensions = input_dimensions
        self.action_dimensions = action_dimensions
        self.learning_rate = learning_rate
        self.fc1 = torch.nn.Linear(in_features=self.input_dimensions, out_features=128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.V = torch.nn.Linear(128, 1)
        self.A = torch.nn.Linear(128, action_dimensions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)


    def forward(self, state):
        # state = torch.Tensor(state)
        l1 = torch.nn.functional.relu(self.fc1(state))
        l2 = torch.nn.functional.relu(self.fc2(l1))
        V = self.V(l2)
        A = self.A(l2)

        return V, A


class DQNAgent:

    def __init__(self, input_dimensions, action_dimensions, memory_size=10000,
                 epsilon=1, epsilon_dec=0.9996, epsilon_end=0.1, learning_rate=0.01, gamma=0.99, batch_size=64, model_name="model"):
        # Environment related
        self.input_dimensions = input_dimensions
        self.action_dimensions = action_dimensions
        # Hyperparameter
        self.epsilon = epsilon
        self.epsilon_dec = epsilon_dec
        self.epsilon_end = epsilon_end
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        # neural network
        self.neural_network = DuelingDeepQNetwork(input_dimensions=self.input_dimensions,
                                           action_dimensions=self.action_dimensions,
                                           learning_rate=self.learning_rate)
        # Experience Replay
        self.experience_replay = PrioritizedExperienceReplay(memory_size)
        # Model Name
        self.model_name = model_name


    def remember(self, state, action, reward, new_state, terminal):
        action_space = [0] * self.action_dimensions
        action_space[action] = 1
        self.experience_replay.add(state, action_space, reward, new_state, terminal)

    def choose_action(self, state):
        if torch.rand(1).item() < self.epsilon:
            rnd_action = torch.randint(0, self.action_dimensions, (1,)).item()
            return rnd_action
        else:
            state = torch.Tensor(state)
            _, advantage = self.neural_network.forward(state)

            return torch.argmax(advantage).item()

    def learn(self):
        if self.experience_replay.contains_more_elements_than_batch_size(self.batch_size):
            self.neural_network.optimizer.zero_grad()
            state_batch, action_batch, reward_batch, new_state_batch, terminal_batch, importance = self.experience_replay.sample_experiences(self.batch_size)
            state_batch = torch.tensor(state_batch, dtype=torch.float)
            new_state_batch = torch.tensor(new_state_batch, dtype=torch.float)
            reward_batch = torch.tensor(reward_batch)
            terminal_batch = torch.tensor(terminal_batch)
            action_batch = torch.tensor(action_batch)
            action_indices = action_batch.argmax(dim=1).tolist()

            V_s, A_s = self.neural_network.forward(state_batch)
            V_s_, A_s_ = self.neural_network.forward(new_state_batch)

            Q_s_a = torch.add(V_s, (
                        A_s - A_s.mean(dim=1, keepdim=True)))  # .gather(1,action_batch.unsqueeze(-1)).squeeze(-1)

            Q_s_a_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

            y = Q_s_a.clone()  # just for same format..
            batch_index = torch.arange(self.batch_size)
            y[batch_index, action_indices] = reward_batch + self.gamma * torch.max(Q_s_a_next, dim=1)[0] * terminal_batch


            # LOSS
            importance = torch.tensor(importance, dtype=float)
            loss = (importance * torch.nn.functional.mse_loss(y, Q_s_a)).mean()
            loss.backward()
            self.neural_network.optimizer.step()
            self.epsilon = self.epsilon * self.epsilon_dec if self.epsilon > self.epsilon_end \
                else self.epsilon_end



    def save(self):
        torch.save(self.neural_network, self.model_name+".pt")

    def load(self):
        try:
            self.neural_network = torch.load(self.model_name+".pt")
            self.neural_network.eval()
            print("Model found")
        except:
            print("No model found")


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = DQNAgent(input_dimensions=4, action_dimensions=2, memory_size=10000,
                     epsilon=1, epsilon_dec=0.98, epsilon_end=0.1, learning_rate=0.01, gamma=0.99, batch_size=64)
    agent.load()
    scores = deque()
    eps_history = []
    num_games = 1000
    score = 0

    for i in range(num_games):
        eps_history.append(agent.epsilon)
        done = False
        state = env.reset()
        score = 0
        while not done:
            action = agent.choose_action(state)
            state_, reward, done, info = env.step(action)
            score += reward
            agent.remember(state, action, reward, state_, done)
            state = state_
            agent.learn()
            if i % 10 == 0:
                env.render()
                agent.save()

        scores.append(score)
        print("%d.Episode:" % i,
              "Current Score:%.2f" % score,
              "Average Score:%.2f"%torch.mean(torch.Tensor(scores)).item(),
              "Current Epsilon:%.2f"%agent.epsilon)