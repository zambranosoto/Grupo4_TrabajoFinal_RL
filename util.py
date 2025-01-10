import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # Verifica las formas de los estados antes de agregarlos al buffer
        # print(f"Push: State shape {state.shape}, Next state shape {next_state.shape}")
        self.buffer.append((
            np.array(state).squeeze(),  # Quitar dimensiones adicionales si las hay
            action,
            reward,
            np.array(next_state).squeeze(),  # Quitar dimensiones adicionales si las hay
            done
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Verifica las formas de los estados al muestrear del buffer
        # print(f"Sample: State shape {np.array(states).shape}, Next state shape {np.array(next_states).shape}")

        return (
            np.array(states),  # Mantener forma consistente
            np.array(actions),
            np.array(rewards),
            np.array(next_states),  # Mantener forma consistente
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

def plot_metrics(rewards, losses, save_path="runs/logs"):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Rewards over Episodes")
    plt.legend()
    plt.savefig(f"{save_path}/rewards.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over Training Steps")
    plt.legend()
    plt.savefig(f"{save_path}/losses.png")
    plt.close()
