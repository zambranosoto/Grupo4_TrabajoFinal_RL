import gymnasium as gym
import ale_py
import torch
from model import DQN, save_model, load_model
from train import train_dqn
from test import evaluate_agent
from config import CONFIG
import os

if __name__ == "__main__":
    os.makedirs("runs/logs", exist_ok=True)
    os.makedirs("runs/models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    env = gym.make("ALE/Seaquest-v5", render_mode="rgb_array")
    # env = gym.make("ALE/Seaquest-v5", render_mode="human")
    action_space = env.action_space.n

    policy_net = DQN(input_channels=3, num_actions=action_space).to(device)
    target_net = DQN(input_channels=3, num_actions=action_space).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    mode = input("Train or Test? (train/test): ").lower()

    if mode == "train":
        rewards, losses = train_dqn(env, policy_net, target_net, CONFIG, device)
        save_model(policy_net, "runs/models/seaquest_dqn_2.pth")
        print("Training complete. Model saved.")

    elif mode == "test":
        model = load_model("runs/models/seaquest_dqn.pth", action_space, device)
        evaluate_agent(env, model, num_episodes=10, device=device, render=True)
