import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from flwr.client import ClientApp, NumPyClient
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models

# Parse arguments
parser = argparse.ArgumentParser(description="Flower")
parser.add_argument(
    "--partition-id",
    type=int,
    choices=[0, 1],
    default=0,
    help="Partition of the dataset (0 or 1). "
    "The dataset is divided into 2 partitions (train and test).",
)
args, _ = parser.parse_known_args()

# Define device

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define data directories
train_path = "/home/saidakca/workspace/Workspace/oct_fl/c1/train"
test_path = "/home/saidakca/workspace/Workspace/oct_fl/c1/test"

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Create data loaders
train_dataset = datasets.ImageFolder(train_path, data_transforms['train'])
test_dataset = datasets.ImageFolder(test_path, data_transforms['test'])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define model
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 4)  # 4 classes
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_labels), np.array(all_preds)

'''

def plot_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()
    plt.close()

def plot_accuracy_loss(history, title, filename):
    plt.figure(figsize=(12, 4))
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Accuracy')
    plt.title(title + ' - Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Loss')
    plt.title(title + ' - Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    # plt.show()
    plt.close()


class FlowerClient(NumPyClient):
    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in model.parameters()]

    def fit(self, parameters, config):
        for model_param, new_param in zip(model.parameters(), parameters):
            model_param.data = torch.tensor(new_param).to(model_param.device)
        history = {'loss': [], 'accuracy': []}
        for epoch in range(3):  # Adjust epochs as needed
            epoch_loss, epoch_acc = train(model, train_loader, criterion, optimizer, device)
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            print(f"Epoch {epoch+1}: Loss {epoch_loss}, Accuracy {epoch_acc}")
        plot_accuracy_loss(history, f'Client {args.partition_id} Training', f'accuracy_loss_client_{args.partition_id}.png')
        return [val.detach().cpu().numpy() for val in model.parameters()], len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        for model_param, new_param in zip(model.parameters(), parameters):
            model_param.data = torch.tensor(new_param).to(model_param.device)
        loss, accuracy, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, list(test_loader.dataset.classes), f'Client {args.partition_id} Confusion Matrix', f'confusion_matrix_client_{args.partition_id}.png')
        report = classification_report(y_true, y_pred, target_names=list(test_loader.dataset.classes), output_dict=True)
        print(f'Client {args.partition_id} Evaluation Report:\n', report)
        return loss, len(test_loader.dataset), {
            "accuracy": accuracy, 
            "f1_score": report['weighted avg']['f1-score'], 
            "precision": report['weighted avg']['precision'], 
            "recall": report['weighted avg']['recall']
        }




def client_fn(cid: str):
    return FlowerClient().to_client()

app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )

'''


def plot_confusion_matrix(cm, labels, title, filename, round_num, text_color='orange'):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    filename = f"{filename}_round_{round_num}.png"  # Add round number to filename
    plt.savefig(filename)
    plt.close()

def plot_accuracy_loss(history, title, filename, round_num):
    plt.figure(figsize=(10, 5))
    epochs = range(1, len(history['loss']) + 1)
    plt.plot(epochs, history['accuracy'], 'b-', label='Accuracy')
    plt.plot(epochs, history['loss'], 'r-', label='Loss')
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    filename = f"{filename}_round_{round_num}.png"  # Add round number to filename
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


class FlowerClient(NumPyClient):
    def __init__(self):
        super().__init__()
        self.round_counter = 0

    def get_parameters(self, config):
        return [val.detach().cpu().numpy() for val in model.parameters()]

    def fit(self, parameters, config):
        for model_param, new_param in zip(model.parameters(), parameters):
            model_param.data = torch.tensor(new_param).to(model_param.device)
        history = {'loss': [], 'accuracy': []}
        self.round_counter += 1  # Increment round counter
        for epoch in range(5):  # Adjust epochs as needed
            epoch_loss, epoch_acc = train(model, train_loader, criterion, optimizer, device)
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            print(f"Epoch {epoch+1}, Round {self.round_counter}: Loss {epoch_loss}, Accuracy {epoch_acc}")
        plot_accuracy_loss(history, f'Client {args.partition_id} Training', f'accuracy_loss_client2_{args.partition_id}', self.round_counter)
        return [val.detach().cpu().numpy() for val in model.parameters()], len(train_loader.dataset), {}

    def evaluate(self, parameters, config):
        for model_param, new_param in zip(model.parameters(), parameters):
            model_param.data = torch.tensor(new_param).to(model_param.device)
        loss, accuracy, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        print(f"Confusion Matrix Predictions: {y_pred}")
        print(f"Confusion Matrix True Labels: {y_true}")
        cm = confusion_matrix(y_true, y_pred)
        plot_confusion_matrix(cm, list(test_loader.dataset.classes), f'Client {args.partition_id} Confusion Matrix', f'confusion_matrix_client2_{args.partition_id}', self.round_counter)
        report = classification_report(y_true, y_pred, target_names=list(test_loader.dataset.classes), output_dict=True)
        print(f'Client {args.partition_id} Evaluation Report:\n', report)
        return loss, len(test_loader.dataset), {
            "accuracy": accuracy, 
            "f1_score": report['weighted avg']['f1-score'], 
            "precision": report['weighted avg']['precision'], 
            "recall": report['weighted avg']['recall']
        }


def client_fn(cid: str):
    return FlowerClient().to_client()

app = ClientApp(
    client_fn=client_fn,
)

if __name__ == "__main__":
    from flwr.client import start_client

    start_client(
        server_address="127.0.0.1:8080",
        client=FlowerClient().to_client(),
    )
