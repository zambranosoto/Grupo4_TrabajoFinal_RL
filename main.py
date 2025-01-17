import gymnasium as gym
import ale_py
import torch
from model import DQN, save_model, load_model
from train import train_dqn
from test import evaluate_agent
from config import CONFIG
import os

if __name__ == "__main__":
    """
    Configura y ejecuta el entrenamiento o prueba del modelo DQN.

    Crea directorios para guardar logs y modelos.
    Inicializa el entorno y los modelos.
    Pregunta al usuario si desea entrenar o probar el modelo.
    """
    os.makedirs("runs/logs", exist_ok=True)
    os.makedirs("runs/models", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # env = gym.make("SeaquestNoFrameskip-v4", render_mode="rgb_array")
    env = gym.make("SeaquestNoFrameskip-v4", render_mode="human")
    action_space = env.action_space.n
    exploration_type = CONFIG.get("exploration_type", "e-greedy")

    policy_net = DQN(input_channels=3, num_actions=action_space, exploration_type=exploration_type).to(device)
    target_net = DQN(input_channels=3, num_actions=action_space, exploration_type=exploration_type).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    mode = input("Train or Test? (train/test): ").lower()

    if mode == "train":
        rewards, losses = train_dqn(env, policy_net, target_net, CONFIG, device)
        save_model(policy_net, f"runs/models/seaquest_dqn_{exploration_type}.pth")
        print("Training complete. Model saved.")

    elif mode == "test":
        model = load_model(f"runs/models/seaquest_dqn_{exploration_type}_2.pth", action_space, device, exploration_type=exploration_type)
        evaluate_agent(env, model, num_episodes=10, device=device, render=True)
