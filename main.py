import gymnasium as gym
import ale_py
from model import DQN, save_model, load_model
from train import train_dqn
from test import evaluate_agent
from config import CONFIG

if __name__ == "__main__":
    env = gym.make("ALE/Seaquest-v5", render_mode="rgb_array")
    action_space = env.action_space.n
    policy_net = DQN(action_space)
    target_net = DQN(action_space)
    target_net.load_state_dict(policy_net.state_dict())

    mode = input("Train or Test? (train/test): ").lower()

    if mode == "train":
        rewards, losses = train_dqn(env, policy_net, target_net, CONFIG)
        save_model(policy_net, "runs/models/seaquest_dqn.pth")
        print("Training complete. Model saved.")

    elif mode == "test":
        model = load_model("runs/models/seaquest_dqn.pth", action_space)
        evaluate_agent(env, model, num_episodes=10, render=True)
