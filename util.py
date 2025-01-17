import random
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import torch

class ReplayBuffer:
    """
    Clase para el buffer de repetición, que almacena experiencias de entrenamiento.

    Atributos:
    ----------
    buffer : deque
        Un buffer de doble extremo (deque) para almacenar las experiencias.

    Métodos:
    --------
    push(state, action, reward, next_state, done):
        Añade una experiencia al buffer.
    sample(batch_size):
        Muestra un lote de experiencias del buffer.
    __len__():
        Retorna el tamaño del buffer.
    """

    def __init__(self, capacity):
        """
        Inicializa el buffer de repetición con una capacidad específica.

        Parámetros:
        -----------
        capacity : int
            La capacidad máxima del buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Añade una experiencia al buffer.

        Parámetros:
        -----------
        state : np.array
            El estado actual.
        action : int
            La acción tomada.
        reward : float
            La recompensa recibida.
        next_state : np.array
            El siguiente estado.
        done : bool
            Indicador de si el episodio ha terminado.
        """
        self.buffer.append((
            np.array(state, dtype=np.float16).squeeze(),
            action,
            reward,
            np.array(next_state, dtype=np.float16).squeeze(),
            done
        ))

    def sample(self, batch_size):
        """
        Muestra un lote de experiencias del buffer.

        Parámetros:
        -----------
        batch_size : int
            El tamaño del lote a muestrear.

        Retorna:
        --------
        tuple
            Un lote de experiencias (estados, acciones, recompensas, siguientes estados, dones).
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convertir a arrays de numpy en float16 para ahorrar memoria
        states = np.array(states, dtype=np.float16)
        actions = np.array(actions, dtype=np.int64)  # torch.long equivale a int64
        rewards = np.array(rewards, dtype=np.float16)
        next_states = np.array(next_states, dtype=np.float16)
        dones = np.array(dones, dtype=np.float16)

        # Convertir a tensores en float16 para la GPU
        states = torch.tensor(states, dtype=torch.float16).to('cuda')
        actions = torch.tensor(actions, dtype=torch.long).to('cuda')
        rewards = torch.tensor(rewards, dtype=torch.float16).to('cuda')
        next_states = torch.tensor(next_states, dtype=torch.float16).to('cuda')
        dones = torch.tensor(dones, dtype=torch.float16).to('cuda')

        # Forzar la recolección de basura
        import gc
        gc.collect()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Retorna el tamaño actual del buffer.

        Retorna:
        --------
        int
            El número de experiencias almacenadas en el buffer.
        """
        return len(self.buffer)

def plot_metrics(rewards, losses, exploration_type, save_path="runs/logs"):
    """
    Grafica las recompensas y pérdidas durante el entrenamiento.

    Parámetros:
    -----------
    rewards : list
        Lista de recompensas acumuladas por episodio.
    losses : list
        Lista de pérdidas acumuladas por paso.
    exploration_type : str
        Tipo de exploración utilizada ("e-greedy" o "noisynet").
    save_path : str
        La ruta donde se guardarán las gráficas.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Reward")
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Rewards over Episodes")
    plt.legend()
    plt.savefig(f"{save_path}/rewards_{exploration_type}_1.png")
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.title("Loss over Training Steps")
    plt.legend()
    plt.savefig(f"{save_path}/losses_{exploration_type}_1.png")
    plt.close()
