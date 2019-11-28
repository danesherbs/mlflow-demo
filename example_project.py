import mlflow
import random


# Can call mlflow.start_run() or wrap in `with` statement.
# New runs are launched under the "Ablation study" experiment.
mlflow.set_experiment("Ablation study")
mlflow.start_run(run_name="M-DGI")

# log_param (optimiser, num layers, ...) are for constants
mlflow.log_param("lr", random.random() * 1e-3)
mlflow.log_param("n_layers", random.randint(100, 1000))
mlflow.log_param("optimiser", "adam")

for epoch in range(100):
    # log_metric logs metrics (acc, recall, ...) are for things that change
    mlflow.log_metric("precision", random.random() * epoch, step=epoch)
    mlflow.log_metric("recall", random.random() * epoch, step=epoch)
