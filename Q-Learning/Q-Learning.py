from __future__ import annotations
import MalmoPython
import json
import logging
import os
import random
import sys
import time
import tkinter as tk
from collections import defaultdict

# ============================================================
# Tabular Q-Learning Agent (Optimized)
# ============================================================

class TabQAgent:
    def __init__(self):
        # --- Hyperparameters ---
        self.alpha = 0.8     # learning rate
        self.gamma = 0.9     # discount factor
        self.epsilon = 0.01  # exploration rate

        # --- Actions ---
        self.actions = [
            "movenorth 1",
            "movesouth 1",
            "movewest 1",
            "moveeast 1"
        ]

        # --- Q-table ---
        self.q_table = defaultdict(lambda: [0.0] * len(self.actions))

        # --- Previous state/action ---
        self.prev_state = None
        self.prev_action = None

        # --- Logging ---
        self.logger = logging.getLogger("TabQAgent")
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        self.logger.handlers = [handler]

        # --- Visualization ---
        self.root = None
        self.canvas = None

    # ============================================================
    # Q-learning updates
    # ============================================================

    def update_q(self, reward: float, current_state: str):
        old_q = self.q_table[self.prev_state][self.prev_action]
        best_next_q = max(self.q_table[current_state])
        new_q = old_q + self.alpha * (reward + self.gamma * best_next_q - old_q)
        self.q_table[self.prev_state][self.prev_action] = new_q

    def update_terminal_q(self, reward: float):
        old_q = self.q_table[self.prev_state][self.prev_action]
        new_q = old_q + self.alpha * (reward - old_q)
        self.q_table[self.prev_state][self.prev_action] = new_q

    # ============================================================
    # Action selection
    # ============================================================

    def choose_action(self, state: str) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, len(self.actions) - 1)

        max_q = max(self.q_table[state])
        best_actions = [
            i for i, q in enumerate(self.q_table[state]) if q == max_q
        ]
        return random.choice(best_actions)

    # ============================================================
    # Acting step
    # ============================================================

    def act(self, world_state, agent_host, reward):
        obs = json.loads(world_state.observations[-1].text)

        if "XPos" not in obs or "ZPos" not in obs:
            return reward

        x, z = int(obs["XPos"]), int(obs["ZPos"])
        current_state = f"{x}:{z}"

        # Update Q-table
        if self.prev_state is not None:
            self.update_q(reward, current_state)

        # Choose and send action
        action_idx = self.choose_action(current_state)
        agent_host.sendCommand(self.actions[action_idx])

        self.prev_state = current_state
        self.prev_action = action_idx

        self.draw_q(x, z)
        return reward

    # ============================================================
    # Main episode loop
    # ============================================================

    def run(self, agent_host):
        total_reward = 0.0
        self.prev_state = None
        self.prev_action = None

        while True:
            world_state = agent_host.getWorldState()
            if not world_state.is_mission_running:
                break

            reward = sum(r.getValue() for r in world_state.rewards)

            if world_state.observations and world_state.observations[-1].text != "{}":
                total_reward += self.act(world_state, agent_host, reward)

            time.sleep(0.05)

        # Terminal update
        if self.prev_state is not None:
            self.update_terminal_q(reward)

        self.draw_q()
        return total_reward

    # ============================================================
    # Visualization
    # ============================================================

    def draw_q(self, curr_x=None, curr_y=None):
        scale = 40
        world_x, world_y = 6, 14

        if self.root is None:
            self.root = tk.Tk()
            self.root.title("Q-table")
            self.canvas = tk.Canvas(
                self.root,
                width=world_x * scale,
                height=world_y * scale,
                bg="black"
            )
            self.canvas.pack()

        self.canvas.delete("all")

        action_offsets = [
            (0.5, 0.1), (0.5, 0.9),
            (0.1, 0.5), (0.9, 0.5)
