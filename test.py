import numpy as np
import torch
import matplotlib.pyplot as plt

def evaluate_agent(env, policy_net, num_episodes, render=False):
    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        while True:
            with torch.no_grad():
                action = torch.argmax(policy_net(torch.FloatTensor(state))).item()

            state, reward, done, _, _ = env.step(action)
            episode_reward += reward

            if render:
                env.render()

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode}: Reward = {episode_reward}")

    avg_reward = np.mean(total_rewards)
    std_reward = np.std(total_rewards)

    print(f"\nEvaluation Results: Average Reward = {avg_reward:.2f}, Std Dev = {std_reward:.2f}")

    # Gráfico de recompensas durante la evaluación
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_episodes), total_rewards, label="Reward per Episode")
    plt.axhline(y=avg_reward, color='r', linestyle='--', label=f"Avg Reward = {avg_reward:.2f}")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.title("Evaluation Rewards")
    plt.legend()
    plt.savefig("runs/logs/evaluation_rewards.png")
    plt.close()
