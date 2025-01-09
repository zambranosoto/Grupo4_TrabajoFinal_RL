import torch.nn as nn
import torch

"""
class DQN(nn.Module):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(84 * 84 * 4, 512),  # Procesamos imágenes de 4 frames.
            nn.ReLU(),
            nn.Linear(512, action_space)  # Salida: acciones posibles.
        )

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))  # Aplanar la entrada.
"""

class DQN(nn.Module):
    def __init__(self, action_space):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        return self.fc2(x)

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, action_space):
    model = DQN(action_space)
    model.load_state_dict(torch.load(path))
    return model
