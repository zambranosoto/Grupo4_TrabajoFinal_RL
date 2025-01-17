import torch.nn as nn
import torch
import math


class NoisyLinear(nn.Module):
    """
    Implementa una capa lineal NoisyNet que añade ruido a los parámetros para mejorar la exploración.

    Atributos:
    ----------
    in_features : int
        Número de características de entrada.
    out_features : int
        Número de características de salida.
    sigma_init : float
        Inicialización del sigma para el ruido.
    weight_mu : nn.Parameter
        Media de los pesos.
    weight_sigma : nn.Parameter
        Sigma de los pesos.
    weight_epsilon : torch.Tensor
        Epsilon para los pesos.
    bias_mu : nn.Parameter
        Media del sesgo.
    bias_sigma : nn.Parameter
        Sigma del sesgo.
    bias_epsilon : torch.Tensor
        Epsilon para el sesgo.

    Métodos:
    --------
    reset_parameters():
        Inicializa los parámetros de la capa.
    reset_noise():
        Resetea el ruido para los pesos y el sesgo.
    forward(input):
        Realiza la propagación hacia adelante de la capa.
    """
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        if bias:
            self.bias_mu = nn.Parameter(torch.empty(out_features))
            self.bias_sigma = nn.Parameter(torch.empty(out_features))
            self.register_buffer('bias_epsilon', torch.empty(out_features))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_sigma', None)
            self.register_buffer('bias_epsilon', None)

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """
        Inicializa los parámetros de la capa.
        """
        mu_range = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.sigma_init)

    def reset_noise(self):
        """
        Resetea el ruido para los pesos y el sesgo.
        """
        self.weight_epsilon.normal_()
        if self.bias_epsilon is not None:
            self.bias_epsilon.normal_()

    def forward(self, input):
        """
        Realiza la propagación hacia adelante de la capa.

        Parámetros:
        -----------
        input : torch.Tensor
            Tensor de entrada.

        Retorna:
        --------
        torch.Tensor
            Tensor de salida después de aplicar la capa lineal con ruido.
        """
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon if self.bias_mu is not None else None
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(input, weight, bias)


class DQN(nn.Module):
    """
    Define la arquitectura de la Red Neuronal Profunda (DQN) utilizada para el entrenamiento del agente.

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
    fc1 : nn.Linear o NoisyLinear
        Primera capa totalmente conectada (con o sin ruido).
    fc2 : nn.Linear o NoisyLinear
        Segunda capa totalmente conectada (con o sin ruido).
    exploration_type : str
        Tipo de exploración a utilizar ("e-greedy" o "noisynet").

    Métodos:
    --------
    forward(x):
        Realiza la propagación hacia adelante de la red neuronal.
    reset_noise():
        Resetea el ruido en las capas lineales NoisyNet.
    """
    def __init__(self, input_channels, num_actions, exploration_type="e-greedy"):
        """
        Inicializa la Red Neuronal Profunda (DQN) con las capas especificadas.

        Parámetros:
        -----------
        input_channels : int
            Número de canales de entrada (por ejemplo, 3 para imágenes RGB).
        num_actions : int
            Número de acciones posibles en el entorno.
        exploration_type : str
            Tipo de exploración a utilizar ("e-greedy" o "noisynet").
        """
        super(DQN, self).__init__()
        self.exploration_type = exploration_type
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1)  # Nueva capa convolucional

        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 210, 160)
            conv_out = self.conv4(self.conv3(self.conv2(self.conv1(dummy_input))))
        if exploration_type == "noisynet":
            self.fc1 = NoisyLinear(conv_out.numel(), 512)
            self.fc2 = NoisyLinear(512, num_actions)
        else:
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

    def reset_noise(self):
        """
        Resetea el ruido en las capas lineales NoisyNet, si se está utilizando NoisyNet.
        """
        if self.exploration_type == "noisynet":
            self.fc1.reset_noise()
            self.fc2.reset_noise()


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

def load_model(path, action_space, device, exploration_type="e-greedy"):
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
    exploration_type : str, opcional
        El tipo de exploración a utilizar ("e-greedy" o "noisynet").

    Retorna:
    --------
    DQN
        El modelo DQN cargado.
    """
    model = DQN(input_channels=3, num_actions=action_space, exploration_type=exploration_type).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model
