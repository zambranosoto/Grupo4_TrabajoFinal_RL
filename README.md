# Informe de Implementación del Modelo DQN en el Entorno SeaquestNoFrameskip-v4

Este informe presenta la implementación de un modelo DQN para entrenar un agente en el entorno ALE/Seaquest-v5 y SeaquestNoFrameskip-v4. Se comparan dos estrategias de exploración: ε-greedy, que selecciona acciones de manera controlada y aleatoria, y NoisyNet, que introduce ruido paramétrico en la red neuronal para fomentar una exploración adaptativa.

## Descripción General

El agente es entrenado con imágenes del juego SeaQuest y estas son procesadas por una red neuronal para estimar recompensas futuras. La evaluación compara ambas estrategias en términos de puntuación, eficiencia de exploración y convergencia. El informe destaca las principales características de cada estrategia con sus respectivos hiperparámetros.

## Objetivo del Informe

El presente informe detalla la aplicación y comparación del desempeño de dos enfoques de exploración en el algoritmo DQN para entrenar un agente que resuelve el juego SeaQuest de Atari. En este entorno, el agente controla un submarino cuya misión es rescatar buzos mientras enfrenta diversos desafíos.

Decidimos usar la arquitectura DQN ya que se basa en el uso de una red neuronal para aproximar los valores Q(s,a), en lugar de una tabla, lo que le permite trabajar eficientemente con estados de alta dimensionalidad [1]. Para entrenar al agente se usaron los entornos que ofrece Gymnasium para SeaQuest. También se implementan y comparan dos estrategias para el algoritmo DQN:
- **NoisyNet**: Introduce ruido paramétrico en la red neuronal para fomentar una exploración adaptativa y más eficiente.
- **ε-greedy**: Combina exploración aleatoria y explotación, equilibrando la búsqueda de nuevas estrategias con el uso de las acciones que maximicen la recompensa esperada.

El objetivo del informe es analizar el impacto de estas estrategias en el aprendizaje y desempeño del agente, considerando métricas como la puntuación alcanzada (recompensa), la función de pérdida, la eficiencia en la exploración y la convergencia del modelo.

## Estructura del Proyecto

El proyecto está estructurado en los siguientes archivos y módulos:

### `model.py`
Define la arquitectura de la Red Neuronal Profunda (DQN) utilizada para el entrenamiento del agente, incluyendo:
- Clase `DQN`
- Clase `NoisyLinear`
- Funciones `save_model` y `load_model`

### `train.py`
Contiene el proceso de entrenamiento del modelo DQN:
- Función `train_dqn`
- Función `compute_loss`

### `main.py`
Coordina la configuración y ejecución del entrenamiento o la prueba del modelo DQN:
- Inicializa el entorno y los modelos
- Permite al usuario seleccionar entre entrenar y probar el modelo

### `test.py`
Maneja la evaluación del modelo DQN entrenado:
- Función `evaluate_agent`

### `config.py`
Define un diccionario de configuración con los hiperparámetros utilizados para el entrenamiento del modelo DQN.

### `util.py`
Proporciona utilidades auxiliares para el entrenamiento y evaluación del modelo:
- Clase `ReplayBuffer`
- Función `plot_metrics`

## Uso

### Entrenamiento del Modelo

Para entrenar el modelo, ejecuta el script principal `main.py` y selecciona la opción `train` cuando se solicite.

```bash
python main.py

Train or Test? (train/test): train
```

### Prueba del Modelo

Para probar el modelo, previamente entrenado, ejecuta el script principal `main.py` y selecciona la opción `test` cuando se solicite.

```bash
python main.py

Train or Test? (train/test): test
```
