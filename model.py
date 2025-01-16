import torch.nn as nn
import torch

class DQN(nn.Module):
    """
    Clase para la Red Neuronal Profunda (DQN) utilizada en el entrenamiento del agente.

    Atributos:
    ----------
    conv1 : nn.Conv2d
        Primera capa convolucional.
    conv2 : nn.Conv2d
        Segunda capa convolucional.
    conv3 : nn.Conv2d
        Tercera capa convolucional.
    conv4 : nn.Conv2d
        Cuarta capa convolucional.
    fc1 : nn.Linear
        Primera capa totalmente conectada.
    fc2 : nn.Linear
        Segunda capa totalmente conectada.

    Métodos:
    --------
    forward(x):
        Realiza la propagación hacia adelante de la red neuronal.
    """

    def __init__(self, input_channels, num_actions):
        """
        Inicializa la Red Neuronal Profunda (DQN) con las capas especificadas.

        Parámetros:
        -----------
        input_channels : int
            Número de canales de entrada (por ejemplo, 3 para imágenes RGB).
        num_actions : int
            Número de acciones posibles en el entorno.
        """
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)  # Nueva capa convolucional

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 210, 160)
            conv_out = self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))
        self.fc1 = nn.Linear(conv_out.numel(), 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        Realiza la propagación hacia adelante de la red neuronal.

        Parámetros:
        -----------
        x : torch.Tensor
            Tensor de entrada que representa el estado del entorno.

        Retorna:
        --------
        torch.Tensor
            Tensor de salida que representa las Q-values para cada acción posible.
        """
        x = nn.ReLU()(self.conv1(x))
        x = nn.ReLU()(self.conv2(x))
        x = nn.ReLU()(self.conv3(x))
        x = nn.ReLU()(self.conv4(x))  # Nueva capa
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

def save_model(model, path):
    """
    Guarda el estado del modelo en el archivo especificado.

    Parámetros:
    -----------
    model : DQN
        El modelo DQN a guardar.
    path : str
        La ruta donde se guardará el archivo del modelo.
    """
    torch.save(model.state_dict(), path)

def load_model(path, action_space, device):
    """
    Carga el estado del modelo desde el archivo especificado.

    Parámetros:
    -----------
    path : str
        La ruta del archivo desde el cual se cargará el modelo.
    action_space : int
        El número de acciones posibles en el entorno.
    device : torch.device
        El dispositivo (CPU o GPU) en el cual se cargará el modelo.

    Retorna:
    --------
    DQN
        El modelo DQN cargado.
    """
    model = DQN(input_channels=3, num_actions=action_space).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model
