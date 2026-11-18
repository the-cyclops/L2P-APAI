local_results_dir = "/content/results-avalanche"
dest_dir = os.path.join(project_path, "results-avalanche")
os.makedirs(local_results_dir, exist_ok=True)
os.makedirs(dest_dir, exist_ok=True)
data_path = "/content"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

set_seed(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- BENCHMARK SPLIT CIFAR-100 (224x224) ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
])

benchmark = SplitCIFAR100(
    n_experiences=CIFAR_NUM_TASKS,
    seed=SEED,
    return_task_id=False,
    dataset_root=data_path,
    train_transform=train_transform,
    eval_transform=eval_transform
)

# --- HELPER EVALUATOR E LOOP ---
def build_run_name(method_name, params, seed=SEED, n_tasks=CIFAR_NUM_TASKS):
    params_str = "_".join(f"{k}{v}" for k, v in params.items())
    return f"{method_name}_{params_str}_t{n_tasks}_s{seed}"

def get_evaluator(run_name, params=None):
    run_dir = os.path.join(local_results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "run_name": run_name,
        "backbone": VIT_NAME,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "num_tasks": CIFAR_NUM_TASKS,
        "seed": SEED,
        "hyperparameters": params or {}
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

    log_file = open(os.path.join(run_dir, "train.log"), "w")

    evaluator = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        timing_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger(log_file), CSVLogger(run_dir)]
    )
    return evaluator, log_file

def run_experiment(strategy, run_name, log_file):
    results = []
    start_time = time.time()
    for i, experience in enumerate(benchmark.train_stream, 1):
        print(f"\n=== [{run_name}] Task {i}/{CIFAR_NUM_TASKS} (trascorso: {(time.time()-start_time)/60:.1f} min) ===")
        strategy.train(experience, num_workers=4)
        res = strategy.eval(benchmark.test_stream, num_workers=4)
        results.append(res)

    log_file.flush()
    log_file.close()

    run_local_dir = os.path.join(local_results_dir, run_name)
    run_dest_dir = os.path.join(dest_dir, run_name)

    json_path = os.path.join(run_local_dir, "final_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    shutil.copytree(run_local_dir, run_dest_dir, dirs_exist_ok=True)
    print(f"Results of {run_name} saved to {run_dest_dir}")
    return results

def show_results(name):
  import os
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns

  base_path = os.path.join(project_path, 'results-avalanche', name)
  matrix_file = os.path.join(base_path, 'accuracy_matrix.csv')

  if not os.path.exists(matrix_file):
      print(f"Cannot find the accuracy matrix at: {matrix_file}")
      print("Creating the matrix...")
      df_raw = pd.read_csv(os.path.join(base_path, 'eval_results.csv'))
      df_pivot = df_raw.pivot(index='training_exp', columns='eval_exp', values='eval_accuracy') * 100
      df_pivot.to_csv(f'{base_path}/accuracy_matrix.csv')
      acc_matrix = df_pivot.to_numpy()
  else:
      acc_matrix = pd.read_csv(matrix_file, index_col=0).to_numpy()

  num_tasks = acc_matrix.shape[0]
  final_avg_accuracy = np.mean(acc_matrix[-1, :])

  forgetting_per_task = []
  for j in range(num_tasks - 1):
      max_past_acc = np.max(acc_matrix[j:-1, j])
      final_acc = acc_matrix[-1, j]
      forgetting = max_past_acc - final_acc
      forgetting_per_task.append(forgetting)

  avg_forgetting = np.mean(forgetting_per_task)

  print(f"--- CONTINUAL LEARNING METRICS ({name}) ---")
  print(f"Final Average Accuracy: {final_avg_accuracy:.2f}%")
  print(f"Average Forgetting:     {avg_forgetting:.2f}%\n")

  sns.set_theme(style="white")
  fig, axes = plt.subplots(1, 2, figsize=(16, 6))

  mask = np.triu(np.ones_like(acc_matrix, dtype=bool), k=1)
  sns.heatmap(acc_matrix, mask=mask, annot=True, fmt=".1f", cmap="Blues",
              vmin=0, vmax=100, cbar_kws={'label': 'Accuracy (%)'}, ax=axes[0])

  axes[0].set_title("Continual Learning Accuracy Evolution", fontsize=14, fontweight='bold')
  axes[0].set_xlabel("Evaluated on Task", fontsize=12)
  axes[0].set_ylabel("Trained up to Task", fontsize=12)
  axes[0].set_xticks(np.arange(num_tasks) + 0.5, [f"T{i+1}" for i in range(num_tasks)])
  axes[0].set_yticks(np.arange(num_tasks) + 0.5, [f"T{i+1}" for i in range(num_tasks)], rotation=0)

  tasks = [f"Task {i+1}" for i in range(num_tasks - 1)]
  colors = ['#e74c3c' if f > 10 else '#2ecc71' for f in forgetting_per_task]

  axes[1].bar(tasks, forgetting_per_task, color=colors, edgecolor='black')
  axes[1].axhline(avg_forgetting, color='red', linestyle='--', linewidth=2, label=f'Avg Forgetting: {avg_forgetting:.2f}%')

  axes[1].set_title("Catastrophic Forgetting per Task", fontsize=14, fontweight='bold')
  axes[1].set_ylabel("Accuracy Drop (%)", fontsize=12)
  axes[1].legend()
  axes[1].grid(axis='y', linestyle='--', alpha=0.7)
  plt.xticks(rotation=45)

  plt.tight_layout()
  plot_path = os.path.join(base_path, 'metrics_visualization.png')
  plt.savefig(plot_path, dpi=300, bbox_inches='tight')
  print(f"Visualization saved to: {plot_path}")
  plt.show()
