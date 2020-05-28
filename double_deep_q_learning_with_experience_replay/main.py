from collections import deque
import torch
import gym


class ExperienceReplay:

    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.state_memory = deque(maxlen=self.memory_size)
        self.action_memory = deque(maxlen=self.memory_size)
        self.reward_memory = deque(maxlen=self.memory_size)
        self.new_state_memory = deque(maxlen=self.memory_size)
        self.terminal_memory = deque(maxlen=self.memory_size)

    def add(self, state, action, reward, new_state, terminal):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.new_state_memory.append(new_state)
        self.terminal_memory.append(1-terminal)

    def contains_more_elements_than_batch_size(self, batch_size):
        return len(self.state_memory) > batch_size

    def sample_experiences(self, batch_size):

        random_indizes = torch.randperm(len(self.state_memory))[
                         :batch_size].tolist()  # torch.randint(0, len(self.memory), (batch_size,)).tolist()
        return [self.state_memory[i] for i in random_indizes], \
               [self.action_memory[i] for i in random_indizes], \
               [self.reward_memory[i] for i in random_indizes], \
               [self.new_state_memory[i] for i in random_indizes], \
               [self.terminal_memory[i] for i in random_indizes]


class DeepQNetwork(torch.nn.Module):

    def __init__(self, input_dimensions, action_dimensions, learning_rate):
        super(DeepQNetwork, self).__init__()
        self.input_dimensions = input_dimensions
        self.action_dimensions = action_dimensions
        self.learning_rate = learning_rate
        self.fc1 = torch.nn.Linear(in_features=self.input_dimensions, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=64)
        self.fc3 = torch.nn.Linear(in_features=64, out_features=self.action_dimensions)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = torch.nn.MSELoss()


    def forward(self, state):
        #state = torch.Tensor(state)
        t = torch.nn.functional.relu(self.fc1(state))
        t = torch.nn.functional.relu(self.fc2(t))
        actions = self.fc3(t)
        return actions


class DQNAgent:

    def __init__(self, input_dimensions, action_dimensions, memory_size=10000,
                 epsilon=1, epsilon_dec=0.9996, epsilon_end=0.05, learning_rate=0.01, gamma=0.99,
                 batch_size=64, replacing_counter=5, model_name="model"):
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
        self.replacing_counter = replacing_counter
        self.replace_index = 1
        # neural network
        self.neural_network = DeepQNetwork(input_dimensions=self.input_dimensions,
                                           action_dimensions=self.action_dimensions,
                                           learning_rate=self.learning_rate)
        self.neural_network_next = DeepQNetwork(input_dimensions=self.input_dimensions,
                                           action_dimensions=self.action_dimensions,
                                           learning_rate=self.learning_rate)
        self.neural_network_next.load_state_dict(self.neural_network.state_dict())
        # Experience Replay
        self.experience_replay = ExperienceReplay(memory_size)
        # Model Name
        self.model_name = model_name

    def replace_target_network(self):
        if self.replacing_counter % self.replace_index == 0:
            self.neural_network_next.load_state_dict(self.neural_network.state_dict())
            self.replace_index = 1


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
            actions = self.neural_network.forward(state)
            return actions.argmax().item()

    def learn(self):
        if self.experience_replay.contains_more_elements_than_batch_size(self.batch_size):
            self.replace_target_network()
            self.neural_network.optimizer.zero_grad()
            state_batch, action_batch, reward_batch, new_state_batch, terminal_batch = self.experience_replay.sample_experiences(self.batch_size)
            state_batch = torch.tensor(state_batch, dtype=torch.float)
            new_state_batch = torch.tensor(new_state_batch, dtype=torch.float)
            reward_batch = torch.tensor(reward_batch)
            terminal_batch = torch.tensor(terminal_batch)
            action_batch = torch.tensor(action_batch)
            action_indices = action_batch.argmax(dim=1).tolist()
            Q_s_a = self.neural_network.forward(state_batch)
            Q_s_a_next = self.neural_network_next.forward(new_state_batch)
            y = Q_s_a.clone() # just for same format..
            batch_index = torch.arange(self.batch_size)
            y[batch_index, action_indices] = reward_batch + self.gamma * torch.max(Q_s_a_next, dim=1)[0] * terminal_batch
            # Reduce the mean squared error loss of y - Q_s_a
            loss = self.neural_network.loss(y, Q_s_a)
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
            self.neural_network_next = torch.load(self.model_name+".pt")
            self.neural_network_next.eval()
            print("Model found")
        except:
            print("No model found")


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = DQNAgent(input_dimensions=4, action_dimensions=2, memory_size=10000,
                     epsilon=1, epsilon_dec=0.996, epsilon_end=0.01, learning_rate=0.003, gamma=0.99, batch_size=64)
    #agent.load()
    scores = deque()
    eps_history = []
    num_games = 2000
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
            #env.render()
            agent.remember(state, action, reward, state_, done)
            state = state_
            agent.learn()
            if i % 10 == 0:
                agent.save()

        scores.append(score)
        print("%d.Episode:" % i,
              "Current Score:%.2f" % score,
              "Average Score:%.2f"%torch.mean(torch.Tensor(scores)).item(),
              "Current Epsilon:%.2f"%agent.epsilon)
