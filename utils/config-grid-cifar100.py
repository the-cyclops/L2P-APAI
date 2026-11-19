# --- SETTING GRID SEARCH ---
BATCH_SIZE = 64
VIT_NAME = "vit_base_patch16_224"
SEED = 420                         # Seed dedicato
CIFAR_NUM_CLASSES = 100
CIFAR_CLASSES_PER_TASK = 10
CIFAR_NUM_TASKS = 10               # Resta 10 per mantenere 10 classi per task
GRID_TASKS = 3                     # Esegue solo i primi 3 task dello stream
EPOCHS = 5
