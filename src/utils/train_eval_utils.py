import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from models.coatnet import MyCoAtNet
from models.convmixer import ConvMixer
from models.swin_transformer import SwinTransformer

MODELS = {
    "coatnet": MyCoAtNet,
    "convmixer": ConvMixer,
    "swintransformer": SwinTransformer,
}

def separate_parameters(model):
    parameters_decay = set()
    parameters_no_decay = set()
    modules_weight_decay = (torch.nn.Linear, torch.nn.Conv2d)
    modules_no_weight_decay = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)

    for m_name, m in model.named_modules():
        for param_name, param in m.named_parameters():
            full_param_name = f"{m_name}.{param_name}" if m_name else param_name

            if isinstance(m, modules_no_weight_decay):
                parameters_no_decay.add(full_param_name)
            elif param_name.endswith("bias"):
                parameters_no_decay.add(full_param_name)
            elif isinstance(m, modules_weight_decay):
                parameters_decay.add(full_param_name)

    # sanity check
    assert len(parameters_decay & parameters_no_decay) == 0
    assert len(parameters_decay) + len(parameters_no_decay) == len(list(model.parameters()))

    return parameters_decay, parameters_no_decay

def init_linear(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None: torch.nn.init.zeros_(m.bias)


def save_confusion_matrix(true_labels, predicted_labels, filename, classes):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes )
    fig, ax = plt.subplots(figsize=(20, 20))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=9)  # Default fontsize is 10
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
    plt.plot(df["Train 1st Accuracy"], label="Train 1st Accuracy")
    plt.plot(df["Val 1st Accuracy"], label="Val 1st Accuracy")
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
    running_accuracies = [0.0] * 6
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


def evaluate_model(device, model, criterion, test_loader, save_cm=False, cm_path=None, class_encoding=None):
    model.eval()
    running_loss = 0.0
    running_accuracies = [0.0] * 6
    predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            predictions.append(outputs)
            true_labels.append(labels)
            running_loss += loss.item() * inputs.size(0)
            for i, k in enumerate([1, 2, 3, 4, 5, 10]):
                running_accuracies[i] += calculate_top_k_accuracy(outputs, labels, k) * inputs.size(0)

    if save_cm:
        predictions = torch.cat(predictions)
        true_labels = torch.cat(true_labels)
        predictions = torch.argmax(predictions, dim=1)
        true_labels = true_labels.cpu().numpy()
        true_classes = np.unique(true_labels)
        predictions = predictions.cpu().numpy()
        pred_classes = np.unique(predictions)
        classes = np.union1d(true_classes, pred_classes)
        inv_class_encoding = {v: k for k, v in class_encoding.items()}
        encoded_classes = [inv_class_encoding.get(cls) for cls in classes]
        save_confusion_matrix(true_labels, predictions, cm_path, classes=encoded_classes)

    loss = running_loss / len(test_loader.dataset)
    accuracies = [acc / len(test_loader.dataset) for acc in running_accuracies]
    return [loss] + accuracies


def evaluate_test(model_name, model_params, class_encoding, dataloader, path, checkpoint_folder, checkpoint_name, device, criterion, dataset):
    model = MODELS[model_name](num_classes=len(class_encoding), **model_params)
    checkpoint = torch.load(f"{checkpoint_folder}/{checkpoint_name}")
    model.load_state_dict(checkpoint)
    model.to(device)

    loss, acc1, acc2, acc3, acc4, acc5, acc10 = evaluate_model(
        device,
        model,
        criterion,
        dataloader,
        save_cm=True,
        cm_path=f"{path}/confusion_matrix_{dataset}.png",
        class_encoding=class_encoding,
    )
    print(f"Dataset {dataset} - Test Loss: {loss:.4f}, Test 1st Accuracy: {acc1:.4f}")

    return loss, acc1, acc2, acc3, acc4, acc5, acc10
