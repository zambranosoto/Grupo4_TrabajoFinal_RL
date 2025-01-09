import torch
import torch.optim as optim
import torch.nn.functional as F
from util import ReplayBuffer
from util import plot_metrics
import random

def train_dqn(env, policy_net, target_net, config):
    optimizer = optim.Adam(policy_net.parameters(), lr=config["lr"])
    buffer = ReplayBuffer(config["buffer_size"])
    rewards, losses = [], []

    for episode in range(config["episodes"]):
        state, _ = env.reset()
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
                    action = torch.argmax(policy_net(torch.FloatTensor(state))).item()

            next_state, reward, done, _, _ = env.step(action)
            buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            if len(buffer) >= config["batch_size"]:
                batch = buffer.sample(config["batch_size"])
                loss = compute_loss(policy_net, target_net, batch, config["gamma"])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            if done:
                break

        rewards.append(episode_reward)

        if episode % config["target_update"] == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}: Reward = {episode_reward}, Epsilon = {epsilon:.3f}")

    # Generar gráficos al finalizar el entrenamiento
    plot_metrics(rewards, losses, save_path="runs/logs")

    return rewards, losses


def compute_loss(policy_net, target_net, batch, gamma):
    states, actions, rewards, next_states, dones = batch
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)

    q_values = policy_net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        max_next_q_values = target_net(next_states).max(1)[0]
        target_q_values = rewards + gamma * max_next_q_values * (1 - dones)
