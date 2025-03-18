import torch
import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
import random
import numpy as np

from utils.data_loading_utils import AudioDataset
from utils.train_eval_utils import MODELS, train_epoch, evaluate_test, validate_epoch, plot_loss_and_acc, load_config

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

SCHEDULERS = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
}

DATA_DIR = '../data/final_keystrokes'


def main(args):
    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    config = load_config(args.config)
    print(config)

    train_dataset = AudioDataset(DATA_DIR + '/train', transform_aug=config["transform_aug"])
    val_dataset = AudioDataset(DATA_DIR + '/val', transform_aug=False)
    test_dataset = AudioDataset(DATA_DIR + '/test', transform_aug=False)
    num_classes = len(train_dataset.classes)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    model = MODELS[config["model"]](num_classes=num_classes, **config["model_params"])
    model.to(device)
    model_name = config["model"]
    num_epochs = config["num_epochs"]
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = OPTIMIZERS[config["optimizer"]](
        model.parameters(), **config["optimizer_params"]
    )
    scheduler = SCHEDULERS[config["scheduler"]](
        optimizer, **config["scheduler_params"]
    )

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    path = f"{args.interm_results}/{model_name}_lr_{config['optimizer_params']['lr']}_augmentation_{config['transform_aug']}_optimizer_{config['optimizer']}_scheduler_{config['scheduler']}"
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/config.json", "w") as f:
        json.dump(config, f)

    columns = ["Train Loss", "Train 1st Accuracy", "Train 2nd Accuracy", "Train 3rd Accuracy",
               "Train 4th Accuracy", "Train 5th Accuracy", "Train 10th Accuracy", "Val Loss",
               "Val 1st Accuracy", "Val 2nd Accuracy", "Val 3rd Accuracy", "Val 4th Accuracy",
               "Val 5th Accuracy", "Val 10th Accuracy"]
    df = pd.DataFrame(columns=columns)

    stop_counter = 0
    max_no_improvement = config["max_no_improvement"]
    best_val_acc = 0

    for epoch in tqdm(num_epochs):
        loss, acc1, acc2, acc3, acc4, acc5, acc10 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5, val_acc10 = validate_epoch(
            model, val_loader, criterion, device
        )
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            stop_counter = 0
            checkpoint_name = f"{epoch}_val_acc_{val_acc1:.3f}.pt"
            torch.save(model.state_dict(), path + "/" + checkpoint_name)
        else:
            stop_counter += 1

        if stop_counter > max_no_improvement:
            break

        scheduler.step()

        df.loc[epoch] = [loss, acc1, acc2, acc3, acc4, acc5, acc10,
                         val_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5, val_acc10]
        print(f"Epoch {epoch + 1}:")
        print(f"Loss: Train={loss:.3f}, Val={val_loss:.3f}")
        print(f"Accuracies: acc1={acc1:.3f}, val_acc1={val_acc1:.3f}, acc3={acc3:.3f}, val_acc3={val_acc3:.3f}, "
              f"acc5={acc5:.3f}, val_acc5={val_acc5:.3f}")
        print(f"Current Best Val 1st Accuracy = {best_val_acc:.3f}, epoch no improvement = {stop_counter}")
        print("/n")

    plot_loss_and_acc(df, f"{path}/loss_acc_plot.png")
    df.to_csv(f"{path}/loss_acc_history.csv")

    evaluate_test(config, test_loader, path, checkpoint_name, device, criterion)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/config.json",
    )
    parser.add_argument(
        '--interm_results',
        type=str,
        required=False,
        default='interim_results/',
    )
    args = parser.parse_args()
    main(args)
