import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.array(state, dtype=np.float16).squeeze(),
            action,
            reward,
            np.array(next_state, dtype=np.float16).squeeze(),
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir a arrays de numpy en float16 para ahorrar memoria
        states = np.array(states, dtype=np.float16)
        actions = np.array(actions, dtype=np.int64)  # torch.long equivale a int64
        rewards = np.array(rewards, dtype=np.float16)
        next_states = np.array(next_states, dtype=np.float16)
        dones = np.array(dones, dtype=np.float16)

        # Convertir a tensores en float16 para la GPU
        states = torch.tensor(states, dtype=torch.float16).to('cuda')
        actions = torch.tensor(actions, dtype=torch.long).to('cuda')
        rewards = torch.tensor(rewards, dtype=torch.float16).to('cuda')
        next_states = torch.tensor(next_states, dtype=torch.float16).to('cuda')
        dones = torch.tensor(dones, dtype=torch.float16).to('cuda')

        # Forzar la recolección de basura
        import gc
        gc.collect()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

def plot_metrics(rewards, losses, save_path="runs/logs"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Rewards over Episodes")
    plt.legend()
    plt.savefig(f"{save_path}/rewards_2.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over Training Steps")
    plt.legend()
    plt.savefig(f"{save_path}/losses_2.png")
    plt.close()
