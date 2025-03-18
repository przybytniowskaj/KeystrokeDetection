import torch
import json
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import seaborn as sns

from ..models import CoAtNet

MODELS = {
    "coatnet": CoAtNet,
}


def save_confusion_matrix(true_labels, predicted_labels, filename, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, cmap="Blues")
    plt.xticks(rotation=45)
    plt.savefig(filename)


def plot_loss_and_acc(df, path):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(df["Train Loss"], label="Train Loss")
    plt.plot(df["Val Loss"], label="Val Loss")
    plt.title("Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(df["Train Accuracy"], label="Train Accuracy")
    plt.plot(df["Val Accuracy"], label="Val Accuracy")
    plt.title("Accuracies")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(path)


def load_config(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    return config


def calculate_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    corrects = torch.sum(preds == labels.data).item()
    return corrects / labels.size(0)


def calculate_top_k_accuracy(outputs, labels, k):
    _, preds = outputs.topk(k, 1, True, True)
    corrects = preds.eq(labels.view(-1, 1).expand_as(preds))
    correct_k = corrects.sum().item()
    return correct_k / labels.size(0)


def train_epoch(device, model, criterion, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    running_accuracies = [0.0] * 10

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        for i, k in enumerate([1, 2, 3, 4, 5, 10]):
            running_accuracies[i] += calculate_top_k_accuracy(outputs, labels, k) * inputs.size(0)

    loss = running_loss / len(train_loader.dataset)
    accuracies = [acc / len(train_loader.dataset) for acc in running_accuracies]
    return [loss] + accuracies


def evaluate_model(device, model, criterion, test_loader, save_cm=False, cm_path=None):
    model.eval()
    running_loss = 0.0
    running_accuracies = [0.0] * 10
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            predictions.append(outputs)
            true_labels.appen(labels)
            running_loss += loss.item() * inputs.size(0)
            for i, k in enumerate([1, 2, 3, 4, 5, 10]):
                running_accuracies[i] += calculate_top_k_accuracy(outputs, labels, k) * inputs.size(0)

    classes = test_loader.dataset.classes
    if save_cm:
        predictions = torch.cat(predictions)
        true_labels = torch.cat(true_labels)
        predictions = torch.argmax(predictions, dim=1)
        true_labels = true_labels.cpu().numpy()
        predictions = predictions.cpu().numpy()
        save_confusion_matrix(true_labels, predictions, cm_path, classes=classes)

    loss = running_loss / len(test_loader.dataset)
    accuracies = [acc / len(test_loader.dataset) for acc in running_accuracies]
    return [loss] + accuracies


def evaluate_test(config, dataloader, path, checkpoint_name, device, criterion):
    model = MODELS[config["model"]](**config["model_params"])
    checkpoint = torch.load(f"{path}/{checkpoint_name}")
    model.load_state_dict(checkpoint)
    model.to(device)

    loss, acc1, acc2, acc3, acc4, acc5, acc10 = evaluate_model(
        device,
        model,
        criterion,
        dataloader,
        save_cm=True,
        cm_path=f"{path}/confusion_matrix.png",
    )
    print(f"Test Loss: {loss:.4f}, Test 1st Accuracy: {acc1:.4f}")
    with open(f"{path}/test_results.txt", "w") as f:
        f.write(f"Test Loss: {loss:.4f}, 1st Accuracy: {acc1:.4f}, 2nd Accuracy: {acc2:.4f}, ",
                f"3rd Accuracy: {acc3:.4f}, 4th Accuracy: {acc4:.4f}, 5th Accuracy: {acc5:.4f}, 10th Accuracy: {acc10:.4f}")
