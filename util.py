import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

def plot_metrics(rewards, losses, save_path="runs/logs"):
    # Recompensas por episodio
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Rewards over Episodes")
    plt.legend()
    plt.savefig(f"{save_path}/rewards.png")
    plt.close()

    # Pérdidas durante el entrenamiento
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over Training Steps")
    plt.legend()
    plt.savefig(f"{save_path}/losses.png")
    plt.close()