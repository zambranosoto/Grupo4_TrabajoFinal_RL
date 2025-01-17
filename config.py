CONFIG = {
    "lr": 1e-4,                      # Tasa de aprendizaje para el optimizador Adam
    "gamma": 0.99,                   # Factor de descuento para la actualización de Q-values
    "epsilon_start": 1.0,            # Valor inicial de epsilon para la política epsilon-greedy
    "epsilon_min": 0.1,              # Valor mínimo de epsilon para la política epsilon-greedy
    "epsilon_decay": 1000,           # Tasa de decaimiento de epsilon para la política epsilon-greedy
    "episodes": 200,                 # Número total de episodios de entrenamiento
    "buffer_size": 50000,            # Tamaño máximo del buffer de repetición
    "batch_size": 16,                # Tamaño del lote para muestreo del buffer de repetición
    "target_update": 10,             # Frecuencia de actualización del modelo objetivo (en episodios)
    "max_steps": 10000,              # Número máximo de pasos por episodio
    "exploration_type": "e-greedy"   # Tipo de exploración a utilizar: "e-greedy" o "noisynet"
    # "exploration_type": "noisynet" # Alternativa para el tipo de exploración
}
