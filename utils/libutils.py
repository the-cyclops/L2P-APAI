import os
import json
import time
import shutil
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import gc
import itertools

from avalanche.benchmarks.classic import SplitCIFAR100
from avalanche.training.plugins import EvaluationPlugin
from avalanche.evaluation.metrics import (
    accuracy_metrics,
    loss_metrics,
    forgetting_metrics,
    timing_metrics,
)
from avalanche.logging import InteractiveLogger, TextLogger, CSVLogger

# Import constants from config.py (SEED, CIFAR_NUM_TASKS, VIT_NAME, BATCH_SIZE, EPOCHS)
from config_cifar100 import *


def setup_environment(project_path, seed=SEED, local_dir="/content/results-avalanche", data_path="/content"):
    """Initialize random seeds, device, local/Drive directories, and return a paths dictionary."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    dest_dir = os.path.join(project_path, "results-avalanche")
    os.makedirs(local_dir, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Environment ready | Device: {device} | Seed: {seed}")

    return {
        "local_results_dir": local_dir,
        "dest_dir": dest_dir,
        "data_path": data_path,
        "device": device,
    }


def get_cifar100_benchmark(data_path="/content", n_tasks=CIFAR_NUM_TASKS, seed=SEED):
    """Return the SplitCIFAR100 benchmark (224x224) configured for ViT."""
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

    return SplitCIFAR100(
        n_experiences=n_tasks,
        seed=seed,
        return_task_id=False,
        dataset_root=data_path,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )


def build_run_name(method_name, params, seed=SEED, n_tasks=CIFAR_NUM_TASKS):
    """Build a standardized run name from the method name, hyperparameters, tasks, and seed."""
    params_str = "_".join(f"{k}{v}" for k, v in params.items())
    return f"{method_name}_{params_str}_t{n_tasks}_s{seed}"


def get_evaluator(run_name, local_results_dir="/content/results-avalanche", params=None, n_tasks=CIFAR_NUM_TASKS):
    """Save run configuration to JSON and return an Avalanche EvaluationPlugin and log file."""
    run_dir = os.path.join(local_results_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    config = {
        "run_name": run_name,
        "backbone": VIT_NAME,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "num_tasks": n_tasks,
        "seed": SEED,
        "hyperparameters": params or {},
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4, default=str)

    log_file = open(os.path.join(run_dir, "train.log"), "w")

    evaluator = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        timing_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger(log_file), CSVLogger(run_dir)],
    )
    return evaluator, log_file

def run_experiment(
    strategy,
    benchmark,
    run_name,
    log_file,
    project_path,
    local_results_dir,
    max_tasks=None,
):
    results = []
    start_time = time.time()
    n_tasks = max_tasks if max_tasks is not None else len(benchmark.train_stream)

    # cut stream on first n_tasks
    train_stream = itertools.islice(benchmark.train_stream, n_tasks)
    test_stream = list(itertools.islice(benchmark.test_stream, n_tasks))

    for i, experience in enumerate(train_stream, 1):
        print(f"\n=== [{run_name}] Task {i}/{n_tasks} (elapsed: {(time.time() - start_time) / 60:.1f} min) ===")
        strategy.train(experience, num_workers=4)
        res = strategy.eval(test_stream, num_workers=4)
        results.append(res)

    total_minutes = round((time.time() - start_time) / 60, 2)
    print(f"\n=== [{run_name}] COMPLETED in {total_minutes} min ===")
    output_data = {
        "total_time_minutes": total_minutes,
        "task_evaluations": results,
    }

    log_file.flush()
    log_file.close()

    run_local_dir = os.path.join(local_results_dir, run_name)
    run_dest_dir = os.path.join(project_path, "results-avalanche", run_name)

    json_path = os.path.join(run_local_dir, "final_metrics.json")
    with open(json_path, "w") as f:
        json.dump(output_data, f, indent=4, default=str)

    shutil.copytree(run_local_dir, run_dest_dir, dirs_exist_ok=True)
    print(f"Results of {run_name} saved to {run_dest_dir}")
    return results


def show_results(name, project_path):
    """Compute final accuracy and forgetting metrics and plot the accuracy heatmap and forgetting bar chart."""
    base_path = os.path.join(project_path, "results-avalanche", name)
    matrix_file = os.path.join(base_path, "accuracy_matrix.csv")

    if not os.path.exists(matrix_file):
        print(f"Cannot find the accuracy matrix at: {matrix_file}")
        print("Creating the matrix...")
        df_raw = pd.read_csv(os.path.join(base_path, "eval_results.csv"))
        df_pivot = df_raw.pivot(index="training_exp", columns="eval_exp", values="eval_accuracy") * 100
        df_pivot.to_csv(matrix_file)
        acc_matrix = df_pivot.to_numpy()
    else:
        acc_matrix = pd.read_csv(matrix_file, index_col=0).to_numpy()

    # Keep only rows that have at least one valid evaluation (handles interrupted runs safely)
    valid_rows = ~np.all(np.isnan(acc_matrix), axis=1)
    acc_matrix = acc_matrix[valid_rows]

    num_tasks = acc_matrix.shape[0]
    last_row_idx = num_tasks - 1
    final_avg_accuracy = float(np.nanmean(acc_matrix[last_row_idx, :last_row_idx + 1]))

    forgetting_per_task = []
    for j in range(last_row_idx):
        past_accs = acc_matrix[j:last_row_idx, j]
        max_past_acc = float(np.nanmax(past_accs)) if not np.all(np.isnan(past_accs)) else 0.0
        final_acc = float(acc_matrix[last_row_idx, j])
        forgetting_per_task.append(max_past_acc - final_acc)

    avg_forgetting = float(np.nanmean(forgetting_per_task)) if len(forgetting_per_task) > 0 else 0.0

    print(f"--- CONTINUAL LEARNING METRICS ({name}) ---")
    print(f"Final Average Accuracy: {final_avg_accuracy:.2f}%")
    print(f"Average Forgetting:     {avg_forgetting:.2f}%\n")

    base, t_part, s_part = name.rsplit('_', 2)
    parts = [p for p in base.split('_') if p]  # gestisce anche il caso di params vuoto
    
    method = parts[0].upper()
    raw_params = parts[1:]
    formatted_params = [
      re.sub(r'^([a-zA-Z]+)([-+]?\d*\.?\d+.*)$', r'\1=\2', p)
      for p in raw_params
    ]
    params_display = ", ".join(formatted_params) if formatted_params else "no params"

    sns.set_theme(style="white")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    fig.suptitle(
      f"{method} ({params_display})  |  Tasks: {t_part[1:]}  |  Seed: {s_part[1:]}\n",
      fontsize=14, fontweight='bold'
    )

    mask = np.triu(np.ones_like(acc_matrix, dtype=bool), k=1)
    sns.heatmap(
        acc_matrix,
        mask=mask,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Accuracy (%)"},
        ax=axes[0],
    )

    axes[0].text(
        0.95, 0.95,
        f"Final Avg Accuracy:\n{final_avg_accuracy:.2f}%",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight='bold',
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#eaf2f8', edgecolor='#2980b9', linewidth=1.5)
    )

    axes[0].set_title(
        f"Accuracy Evolution",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].set_xlabel("Evaluated on Task", fontsize=12)
    axes[0].set_ylabel("Trained up to Task", fontsize=12)
    axes[0].set_xticks(np.arange(acc_matrix.shape[1]) + 0.5, [f"T{i+1}" for i in range(acc_matrix.shape[1])])
    axes[0].set_yticks(np.arange(num_tasks) + 0.5, [f"T{i+1}" for i in range(num_tasks)], rotation=0)

    # Right subplot: Catastrophic Forgetting Bar Chart
    if len(forgetting_per_task) > 0:
        tasks = [f"Task {i+1}" for i in range(len(forgetting_per_task))]
        colors = ["#e74c3c" if f > 10 else "#2ecc71" for f in forgetting_per_task]

        axes[1].bar(tasks, forgetting_per_task, color=colors, edgecolor="black")
        axes[1].axhline(
            avg_forgetting,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Avg Forgetting: {avg_forgetting:.2f}%",
        )
        axes[1].legend()
        axes[1].tick_params(axis="x", rotation=45)
    else:
        axes[1].text(
            0.5, 0.5,
            "Need at least 2 completed tasks\nto compute forgetting",
            ha="center", va="center", fontsize=12, color="gray",
            transform=axes[1].transAxes,
        )

    axes[1].set_title(
        f"Catastrophic Forgetting per Task ",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_ylabel("Accuracy Drop (%)", fontsize=12)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = os.path.join(base_path, "metrics_visualization.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to: {plot_path}")
    plt.show()


def run_grid_search(
    method_name,
    param_grid,
    build_strategy_fn,
    benchmark,
    project_path,
    local_results_dir,
    max_tasks,
    seed,
    show_plots = True
):
    """Run an exhaustive grid search over param_grid, skipping completed runs and cleaning GPU VRAM between runs."""
    dest_dir = os.path.join(project_path, "results-avalanche")
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    total_runs = len(combinations)

    for idx, params in enumerate(combinations, 1):
        base_name = build_run_name(method_name, params, seed, max_tasks)
        run_path = os.path.join(f"grid-{method_name}", base_name)
        print(f"\n>>> [{method_name.upper()}] Run {idx}/{total_runs}: {run_path}")

        if os.path.exists(os.path.join(dest_dir, run_path, "final_metrics.json")):
            print("[SKIP] Already completed on Drive.")
            continue

        strategy = None
        evaluator = None
        extra_objs = None
        log_file = None

        try:
            setup_environment(project_path, seed, local_results_dir)
            evaluator, log_file = get_evaluator(
                run_path, local_results_dir, params, max_tasks
            )
            strategy, extra_objs = build_strategy_fn(params, evaluator)

            run_experiment(
                strategy,
                benchmark,
                run_path,
                log_file,
                project_path,
                local_results_dir,
                max_tasks,
            )
            if show_plots:
                show_results(run_path, project_path)

        finally:
            if log_file and not log_file.closed:
                log_file.flush()
                log_file.close()

            if isinstance(extra_objs, (list, dict, set)):
                extra_objs.clear()
            if evaluator is not None and hasattr(evaluator, "loggers"):
                evaluator.loggers.clear()

            del strategy, extra_objs, evaluator
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()