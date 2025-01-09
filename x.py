import gymnasium as gym
import ale_py

env = gym.make("ALE/Seaquest-v5", render_mode="human")
env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Acción aleatoria
    env.step(action)
env.close()
