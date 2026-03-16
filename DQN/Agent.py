import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# =========================================================
# Replay Memory
# =========================================================

Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done"),
)


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.memory.append(
            Transition(state, action, reward, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.memory)


# =========================================================
# CNN Q-Network
# =========================================================

class Brain(nn.Module):
    def __init__(self, height, width, nb_actions):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),

            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
        )

        # Compute conv output size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 3, height, width)
            conv_out = self.conv(dummy).view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(conv_out, 512),
            nn.ReLU(),
            nn.Linear(512, nb_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# =========================================================
# DQN Agent
# =========================================================

class DQNAgent:
    def __init__(self, hps):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"Running on {self.device}")

        # Networks
        self.policy_net = Brain(
            hps.height, hps.width, hps.nb_actions
        ).to(self.device)

        self.target_net = Brain(
            hps.height, hps.width, hps.nb_actions
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Training
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=hps.learning_rate,
        )

        self.memory = ReplayMemory(hps.memory_capacity)

        self.gamma = hps.gamma
        self.batch_size = hps.batch_size

        # Epsilon-greedy
        self.epsilon = hps.max_epsilon
        self.epsilon_min = hps.min_epsilon
        self.epsilon_decay = hps.decreasing_rate

        self.nb_actions = hps.nb_actions
        self.steps = 0

        self.loss_fn = nn.SmoothL1Loss()

    # -----------------------------------------------------

    def select_action(self, state):

        if random.random() < self.epsilon:
            return random.randrange(self.nb_actions)

        state = (
            torch.tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            q_values = self.policy_net(state)

        return int(q_values.argmax().item())

    # -----------------------------------------------------

    def store(self, s, a, r, s_next, done):
        self.memory.add(s, a, r, s_next, done)

    # -----------------------------------------------------

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return None

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*transitions)

        state = torch.tensor(
            np.array(batch.state), dtype=torch.float32
        ).to(self.device)

        action = torch.tensor(batch.action).unsqueeze(1).to(self.device)
        reward = torch.tensor(batch.reward).to(self.device)

        next_state = torch.tensor(
            np.array(batch.next_state), dtype=torch.float32
        ).to(self.device)

        done = torch.tensor(batch.done, dtype=torch.float32).to(self.device)

        # Q(s,a)
        q_values = self.policy_net(state).gather(1, action).squeeze()

        # max Q'(s')
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0]

        target = reward + (1 - done) * self.gamma * next_q

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), 1.0
        )

        self.optimizer.step()

        # Epsilon decay
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay,
        )

        self.steps += 1

        return loss.item()

    # -----------------------------------------------------

    def update_target(self):
        """Hard update target network"""
        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    # -----------------------------------------------------

    def save(self, path="dqn_model.pt"):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path="dqn_model.pt"):
        self.policy_net.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.update_target()
