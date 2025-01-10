import torch
import torch.optim as optim
import torch.nn as nn
import random
from util import ReplayBuffer, plot_metrics
import numpy as np


def train_dqn(env, policy_net, target_net, config, device):
    optimizer = optim.Adam(policy_net.parameters(), lr=config["lr"])
    buffer = ReplayBuffer(config["buffer_size"])
    rewards, losses = [], []

    for episode in range(config["episodes"]):
        state, _ = env.reset()
        state = np.array(state).squeeze()  # Asegurarse de que no haya dimensiones extra
        # print(f"estado 1: {state.shape}")
        state = torch.FloatTensor(state).to(device).permute(2, 1, 0).unsqueeze(0)  # (B, C, H, W)
        # print(f"estado 2: {state.shape}")
        # state = torch.FloatTensor(state).to(device).permute(1, 2, 0).unsqueeze(0)  # (B, C, H, W)
        # print(f"estado 3: {state.shape}")
        episode_reward = 0

        for t in range(config["max_steps"]):
            epsilon = max(
                config["epsilon_min"],
                config["epsilon_start"] - episode / config["epsilon_decay"]
            )
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = torch.argmax(policy_net(state)).item()

            next_state, reward, done, _, _ = env.step(action)
            # next_state = torch.FloatTensor(next_state).to(device).permute(2, 0, 1).unsqueeze(0)
            next_state = torch.FloatTensor(next_state).to(device).permute(2, 1, 0).unsqueeze(0)
            buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

            # Verifica las formas justo después de agregar al buffer
            # print(f"Train loop: State shape {state.shape}, Next state shape {next_state.shape}")

            state = next_state
            episode_reward += reward

            if len(buffer) >= config["batch_size"]:
                batch = buffer.sample(config["batch_size"])
                loss = compute_loss(policy_net, target_net, batch, config["gamma"], device)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if done:
                print(f"Episode {episode + 1}/{config['episodes']}, Step {t + 1}/{config['max_steps']}: Reward = {episode_reward}, Epsilon = {epsilon:.3f}")
                break

        rewards.append(episode_reward)

        if episode % config["target_update"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # print(f"Episode {episode}: Reward = {episode_reward}, Epsilon = {epsilon:.3f}")
        print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Epsilon = {epsilon:.3f}")

    plot_metrics(rewards, losses, save_path="runs/logs")

    return rewards, losses

def compute_loss(policy_net, target_net, batch, gamma, device):
    states, actions, rewards, next_states, dones = batch
    # print(f"State shape after buffer sampling: {states.shape}")  # Debería ser (batch_size, C, H, W)

    # Convertir a tensores y ajustar dimensiones si es necesario
    states = torch.FloatTensor(states).to(device)
    if states.dim() == 5:  # Si hay una dimensión adicional
        states = states.squeeze(1)
    # states = states.permute(0, 3, 1, 2)  # (B, C, H, W)
    states = states.permute(0, 1, 2, 3)  # (B, C, H, W)

    next_states = torch.FloatTensor(next_states).to(device)
    if next_states.dim() == 5:  # Si hay una dimensión adicional
        next_states = next_states.squeeze(1)
    # next_states = next_states.permute(0, 3, 1, 2)  # (B, C, H, W)
    next_states = next_states.permute(0, 1, 2, 3)  # (B, C, H, W)

    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Q-values actuales
    q_values = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Q-values futuros
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * next_q_values * (1 - dones)

    # Pérdida (Loss)
    loss = nn.MSELoss()(q_values, target_q_values)
    return loss

