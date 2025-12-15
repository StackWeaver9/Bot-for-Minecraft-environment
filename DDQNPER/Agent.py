import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from SumTree import SumTree


# =====================================================
# Replay Memory with Prioritized Experience Replay
# =====================================================

class Memory:
    epsilon = 1e-5
    alpha = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _priority(self, error):
        return (abs(error) + self.epsilon) ** self.alpha

    def add(self, error, sample):
        p = self._priority(error)
        self.tree.add(p, sample)

    def sample(self, batch_size):
        batch = []
        idxs = []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            idx, p, data = self.tree.get(s)
            batch.append(data)
            idxs.append(idx)

        return idxs, batch

    def update(self, idx, error):
        self.tree.update(idx, self._priority(error))


# =====================================================
# CNN Q-Network
# =====================================================

class Brain(nn.Module):
    def __init__(self, height, width, n_actions):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(self._feature_size(height, width), 512)
        self.fc2 = nn.Linear(512, n_actions)

    def _feature_size(self, h, w):
        with torch.no_grad():
            x = torch.zeros(1, 3, h, w)
            x = self._forward_conv(x)
            return x.view(1, -1).size(1)

    def _forward_conv(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 3, 2))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


# =====================================================
# DDQN Agent with PER
# =====================================================

class DDQNPERAgent:
    def __init__(self, hps):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Running on {self.device}")

        self.n_actions = hps.nb_actions
        self.gamma = hps.gamma

        self.model = Brain(hps.height, hps.width, self.n_actions).to(self.device)
        self.target_model = Brain(hps.height, hps.width, self.n_actions).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.optimizer = optim.Adam(self.model.parameters(), lr=hps.learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.memory = Memory(hps.memory_capacity)

        self.batch_size = hps.batch_size
        self.update_target_freq = hps.update_target_frequency

        self.epsilon = hps.max_epsilon
        self.epsilon_min = hps.min_epsilon
        self.epsilon_decay = hps.decreasing_rate

        self.beta = 0.4
        self.beta_increment = 1e-4

        self.steps = 0

    # -------------------------------------------------
    # Action Selection
    # -------------------------------------------------
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.model(state).argmax(dim=1).item()

    # -------------------------------------------------
    # Store Transition
    # -------------------------------------------------
    def observe(self, state, action, reward, next_state):
        self.memory.add(abs(reward), (state, action, reward, next_state))

        self.steps += 1
        self.epsilon = self.epsilon_min + \
            (self.epsilon - self.epsilon_min) * math.exp(-self.epsilon_decay * self.steps)

        if self.steps % self.update_target_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

    # -------------------------------------------------
    # Training Step
    # -------------------------------------------------
    def replay(self):
        if self.memory.tree.n_entries < self.batch_size:
            return

        idxs, batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.stack(states).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)

        # Current Q-values
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN Target
        with torch.no_grad():
            next_actions = self.model(next_states).argmax(dim=1)
            next_q = self.target_model(next_states)
            next_q = next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.gamma * next_q

        # PER weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        errors = (targets - q_values).abs().detach().cpu().numpy()

        loss = self.loss_fn(q_values, targets).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for idx, err in zip(idxs, errors):
            self.memory.update(idx, err)

    # -------------------------------------------------
    # Save / Load
    # -------------------------------------------------
    def save(self, path="ddqn_per"):
        torch.save(self.model.state_dict(), path + "_model.pt")
        torch.save(self.optimizer.state_dict(), path + "_optim.pt")

    def load(self, path="ddqn_per"):
        self.model.load_state_dict(torch.load(path + "_model.pt", map_location=self.device))
        self.target_model.load_state_dict(torch.load(path + "_model.pt", map_location=self.device))
        self.optimizer.load_state_dict(torch.load(path + "_optim.pt", map_location=self.device))
        self.model.eval()
