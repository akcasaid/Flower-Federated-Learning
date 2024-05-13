from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.common import Metrics
import json


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def plot_server_metrics(history, title, filename):
    plt.figure(figsize=(10, 5))
    rounds = range(1, len(history['accuracy']) + 1)
    plt.plot(rounds, history['accuracy'], 'b-', label='Accuracy')
    plt.plot(rounds, history['loss'], 'r-', label='Loss')
    plt.plot(rounds, history['precision'], 'g-', label='Precision')
    plt.plot(rounds, history['recall'], 'c-', label='Recall')
    plt.plot(rounds, history['f1_score'], 'm-', label='F1 Score')
    plt.title(title)
    plt.xlabel('Rounds')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_predictions(y_true_list, y_pred_list, filename):
    with open(filename, 'w') as f:
        json.dump({"y_true": y_true_list, "y_pred": y_pred_list}, f)


strategy = FedAvg(evaluate_metrics_aggregation_fn=weighted_average)


config = ServerConfig(num_rounds=20)

server_metrics = {
    "loss": [],
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1_score": []
}


y_true_list = []
y_pred_list = []

p
app = ServerApp(
    config=config,
    strategy=strategy,
)


if __name__ == "__main__":
    from flwr.server import start_server

    def get_evaluate_fn(server_metrics, y_true_list, y_pred_list):
        def evaluate_fn(round_num, parameters, config):
            # Simulated evaluation logic
            # Replace with actual evaluation results from clients
            y_true = np.random.randint(0, 4, 100)  # Simulated true labels
            y_pred = np.random.randint(0, 4, 100)  # Simulated predictions
            loss = np.random.random()  # Simulated loss
            accuracy = np.mean(y_true == y_pred)  # Simulated accuracy

            precision = precision_score(y_true, y_pred, average='weighted')
            recall = recall_score(y_true, y_pred, average='weighted')
            f1 = f1_score(y_true, y_pred, average='weighted')

            # Append metrics to server_metrics
            server_metrics["loss"].append(loss)
            server_metrics["accuracy"].append(accuracy)
            server_metrics["precision"].append(precision)
            server_metrics["recall"].append(recall)
            server_metrics["f1_score"].append(f1)

            # Append y_true and y_pred to lists
            y_true_list.extend(y_true.tolist())
            y_pred_list.extend(y_pred.tolist())

            return loss, {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1}
        return evaluate_fn

    strategy.evaluate_fn = get_evaluate_fn(server_metrics, y_true_list, y_pred_list)

    start_server(
        server_address="127.0.0.1:8080",
        config=config,
        strategy=strategy,
    )

    plot_server_metrics(server_metrics, "Server Training Metrics", "server_metrics.png")
    save_predictions(y_true_list, y_pred_list, "server_predictions.json")
