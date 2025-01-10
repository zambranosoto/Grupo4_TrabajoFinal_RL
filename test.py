import numpy as np
import torch
import matplotlib.pyplot as plt

def evaluate_agent(env, policy_net, num_episodes, device, render=False):
    total_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = torch.FloatTensor(state).to(device).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        episode_reward = 0

        while True:
            with torch.no_grad():
                action = torch.argmax(policy_net(state)).item()

            next_state, reward, done, _, _ = env.step(action)
            next_state = torch.FloatTensor(next_state).to(device).permute(2, 0, 1).unsqueeze(0)
            state = next_state
            episode_reward += reward

            if render:
                env.render()

            if done:
                break

        total_rewards.append(episode_reward)
        print(f"Episode {episode}: Reward = {episode_reward}")

    avg_reward = np.mean(total_rewards)
    print(f"\nEvaluation Results: Average Reward = {avg_reward:.2f}")

    plt.bar(range(num_episodes), total_rewards)
    plt.axhline(avg_reward, color="r", linestyle="--")
    plt.savefig("runs/logs/evaluation_rewards.png")
    plt.close()
