import torch
import torch.optim as optim
import torch.nn as nn
import random
from torch.amp import autocast, GradScaler
from util import ReplayBuffer, plot_metrics
import numpy as np


def train_dqn(env, policy_net, target_net, config, device):
    """
    Entrena la Red Neuronal Profunda (DQN) en el entorno especificado.

    Parámetros:
    -----------
    env : gym.Env
        El entorno de entrenamiento.
    policy_net : DQN
        El modelo DQN para la política.
    target_net : DQN
        El modelo DQN objetivo.
    config : dict
        Diccionario de configuración con hiperparámetros.
    device : torch.device
        El dispositivo (CPU o GPU) en el cual se entrenará el modelo.

    Retorna:
    --------
    list, list
        Listas con las recompensas y pérdidas durante el entrenamiento.
    """
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
        sum_reward = 0

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

            next_state = torch.FloatTensor(next_state).to(device).permute(2, 1, 0).unsqueeze(0).half()

            # Recompensa por sobrevivir
            if info["lives"] == lives:
                sum_reward += 1
            # Penalización por perder vidas
            if info["lives"] < lives:
                sum_reward -= 10
                lives = info["lives"]
            # Recompensa por rescatar buzo o matar tiburón
            if reward > 0:
                sum_reward += (reward * 1.25)

            buffer.push(state.cpu().numpy(), action, reward, next_state.cpu().numpy(), done)

            state = next_state
            episode_reward += sum_reward

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
                print(
                    f"Episode {episode + 1}/{config['episodes']}, Step {t + 1}/{config['max_steps']}: Reward = {episode_reward}, Epsilon = {epsilon:.3f}")
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
    """
    Calcula la pérdida (loss) de la Red Neuronal Profunda (DQN).

    Parámetros:
    -----------
    policy_net : DQN
        El modelo DQN para la política.
    target_net : DQN
        El modelo DQN objetivo.
    batch : tuple
        Un lote de muestras del buffer de repetición.
    gamma : float
        Factor de descuento para el cálculo de la pérdida.
    device : torch.device
        El dispositivo (CPU o GPU) en el cual se calculará la pérdida.

    Retorna:
    --------
    torch.Tensor
        La pérdida (loss) calculada.
    """
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
