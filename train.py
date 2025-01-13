import torch
import torch.optim as optim
import torch.nn as nn
import random
from torch.amp import autocast, GradScaler
from util import ReplayBuffer, plot_metrics
import numpy as np

def train_dqn(env, policy_net, target_net, config, device):
    scaler = GradScaler()

    optimizer = optim.Adam(policy_net.parameters(), lr=config["lr"])
    buffer = ReplayBuffer(config["buffer_size"])
    rewards, losses = [], []
    lives = 4

    for episode in range(config["episodes"]):
        state, _ = env.reset()
        state = np.array(state, dtype=np.float16).squeeze()
        state = torch.tensor(state, dtype=torch.float16).to(device).permute(2, 1, 0).unsqueeze(0)

        episode_reward = 0

        for t in range(config["max_steps"]):
            epsilon = max(
                config["epsilon_min"],
                config["epsilon_start"] - episode / config["epsilon_decay"]
            )
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                with torch.no_grad(), autocast('cuda'):
                    action = torch.argmax(policy_net(state)).item()

            next_state, reward, done, is_tr, info = env.step(action)
            """
            print(f"reward: {reward}")
            print(f"done: {done}")
            print(f"is_tr: {is_tr}")
            print(f"info: {info}")
            print(f"next state: {next_state}")
            
            print(f"env {dir(env.step(action))}")
            print(f"unwrapped: {dir(env.unwrapped.ale)}")
            print(f"env: {dir(env)}")
            print(f"env.get_wrapper_attr: {dir(env.get_wrapper_attr)}")
            print(f"env.observation_space: {dir(env.get_wrapper_attr)}")
            print(f"env.spec: {dir(env.get_wrapper_attr)}")
            print(f"env.wrapper_spec: {dir(env.get_wrapper_attr)}")
            print(f"env.action_space: {dir(env.get_wrapper_attr)}")
            print(f"env.metadata: {dir(env.metadata)}")
            print(f"env.render: {dir(env.render)}")"""


            next_state = torch.FloatTensor(next_state).to(device).permute(2, 1, 0).unsqueeze(0).half()

            # Implementar recompensas adicionales
            # Recompensa por sobrevivir
            reward += 1
            # Ejemplo: Recompensa por acciones específicas (rescatar buzos)
            if info["lives"] < lives:
                reward -= 10
                lives = info["lives"]
            # Ejemplo: Penalización por colisiones
            # if "hit_enemy" in info and info["hit_enemy"]:
                # reward -= 10

            buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

            state = next_state
            episode_reward += reward

            if len(buffer) >= config["batch_size"]:
                batch = buffer.sample(config["batch_size"])
                loss = compute_loss(policy_net, target_net, batch, config["gamma"], device)

                torch.cuda.empty_cache()  # Liberar memoria antes de la optimización
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                torch.cuda.empty_cache()  # Liberar memoria después de la optimización

                losses.append(loss.item())

            if done:
                print(f"Episode {episode + 1}/{config['episodes']}, Step {t + 1}/{config['max_steps']}: Reward = {episode_reward}, Epsilon = {epsilon:.3f}")
                lives = 4
                break

        rewards.append(episode_reward)

        if episode % config["target_update"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}: Total Reward = {episode_reward}, Epsilon = {epsilon:.3f}")

        torch.cuda.empty_cache()  # Liberar memoria después de cada episodio

    plot_metrics(rewards, losses, save_path="runs/logs")

    return rewards, losses

def compute_loss(policy_net, target_net, batch, gamma, device):
    states, actions, rewards, next_states, dones = batch

    states = states.clone().detach().to(device).half()
    if states.dim() == 5:
        states = states.squeeze(1)

    next_states = next_states.clone().detach().to(device).half()
    if next_states.dim() == 5:
        next_states = next_states.squeeze(1)

    actions = actions.clone().detach().to(device).long()
    rewards = rewards.clone().detach().to(device).half()
    dones = dones.clone().detach().to(device).half()

    with autocast('cuda'):
        q_values = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_q_values = target_net(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, target_q_values)
    return loss
