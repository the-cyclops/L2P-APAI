local_results_dir = "/content/results-avalanche"
dest_dir = os.path.join(project_path, "results-avalanche")
os.makedirs(local_results_dir, exist_ok=True)
os.makedirs(dest_dir, exist_ok=True)
data_path = "/content/data"

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
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
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
def get_evaluator(method_name):
    csv_dir = os.path.join(local_results_dir, method_name)
    os.makedirs(csv_dir, exist_ok=True)
    log_file = open(os.path.join(local_results_dir, f"{method_name}.log"), "w")

    evaluator = EvaluationPlugin(
        accuracy_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=False, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        timing_metrics(experience=True, stream=True),
        loggers=[InteractiveLogger(), TextLogger(log_file), CSVLogger(csv_dir)]
    )
    return evaluator, log_file

def run_experiment(strategy, method_name, log_file):
    results = []
    start_time = time.time()
    for i, experience in enumerate(benchmark.train_stream, 1):
        print(f"\n=== [{method_name.upper()}] Task {i}/{CIFAR_NUM_TASKS} (trascorso: {(time.time()-start_time)/60:.1f} min) ===")
        strategy.train(experience, num_workers=4)
        res = strategy.eval(benchmark.test_stream, num_workers=4)
        results.append(res)

    log_file.flush()
    log_file.close()

    json_path = os.path.join(local_results_dir, method_name, "final_metrics.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=4, default=str)

    shutil.copytree(local_results_dir, dest_dir, dirs_exist_ok=True)
    print(f"Results of {method_name} saved to {dest_dir}")
    return results
