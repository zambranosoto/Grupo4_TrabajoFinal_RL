import torch.nn as nn
import torch

class DQN(nn.Module):
    def __init__(self, input_channels, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Ajusta la dimensión de entrada de fc1 a 64 * 16 * 22
        self.fc1 = nn.Linear(64 * 16 * 22, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        # print(f"Output shape after conv3: {x.shape}")  # Agrega esta línea temporalmente
        x = x.view(x.size(0), -1)  # Aplanar el tensor
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(path, action_space, device):
    model = DQN(input_channels=3, num_actions=action_space).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model
