import torch
import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
import random
import numpy as np
import wandb
import hydra
wandb.login()
from omegaconf import DictConfig, OmegaConf

from src.utils.data_loading_utils import AudioDataset
from src.utils.train_eval_utils import MODELS, train_epoch, evaluate_test, evaluate_model, plot_loss_and_acc, load_config

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "sgd": torch.optim.SGD,
}

SCHEDULERS = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
}

DATA_DIR = './data/final_keystrokes'


@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    train_dataset = AudioDataset(DATA_DIR + '/train', transform_aug=config["transform_aug"], transform=not(config['transform_aug']))
    val_dataset = AudioDataset(DATA_DIR + '/val', transform_aug=False)
    test_dataset = AudioDataset(DATA_DIR + '/test', transform_aug=False)
    num_classes = len(train_dataset.classes)
    class_encoding = train_dataset.class_to_idx

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        # num_workers=config["num_workers"],
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        # num_workers=config["num_workers"],
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        # num_workers=config["num_workers"],
    )

    model = MODELS[cfg.model](num_classes=num_classes, **cfg.model_params)
    model.to(device)
    model_name = cfg.model
    num_epochs = cfg.num_epochs
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = OPTIMIZERS[cfg.optimizer](
        model.parameters(), **cfg.optimizer_params
    )
    scheduler = SCHEDULERS[cfg.scheduler](
        optimizer, **cfg.scheduler_params
    )

    run_name = f"{model_name}_lr_{cfg.optimizer_params.lr}_augmentation_{cfg.transform_aug}_optimizer_{cfg.optimizer}_scheduler_{cfg.scheduler}"
    run = wandb.init(
        entity="przybytniowskaj-warsaw-university-of-technology",
        project="master-thesis",
        name=run_name,
        config=cfg,
    )

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # TODO
    # args.interm_results
    path = f"{'interim_results/'}{run_name}"
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f)

    columns = ["Train Loss", "Train 1st Accuracy", "Train 2nd Accuracy", "Train 3rd Accuracy",
               "Train 4th Accuracy", "Train 5th Accuracy", "Train 10th Accuracy", "Val Loss",
               "Val 1st Accuracy", "Val 2nd Accuracy", "Val 3rd Accuracy", "Val 4th Accuracy",
               "Val 5th Accuracy", "Val 10th Accuracy"]
    df = pd.DataFrame(columns=columns)

    stop_counter = 0
    max_no_improvement = cfg.max_no_improvement
    best_val_acc = 0

    for epoch in tqdm(range(num_epochs)):
        loss, acc1, acc2, acc3, acc4, acc5, acc10 = train_epoch(
            device, model, criterion, optimizer, train_loader
        )
        run.log(
            {
                "train_loss": loss,
                "train_acc1": acc1,
                "train_acc2": acc2,
                "train_acc3": acc3,
                "train_acc4": acc4,
                "train_acc5": acc5,
                "train_acc10": acc10
            }
        )
        val_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5, val_acc10 = evaluate_model(
            device, model, criterion, val_loader
        )
        run.log(
            {
                "val_loss": val_loss,
                "val_acc1": val_acc1,
                "val_acc2": val_acc2,
                "val_acc3": val_acc3,
                "val_acc4": val_acc4,
                "val_acc5": val_acc5,
                "val_acc10": val_acc10
            }
        )
        if epoch == 0:
            checkpoint_folder = path + "/model_checkpoints"
            os.makedirs(checkpoint_folder, exist_ok=True)
            checkpoint_name = f"/{epoch}_val_acc_{val_acc1:.3f}.pt"
            torch.save(model.state_dict(), checkpoint_folder + "/" + checkpoint_name)
        if val_acc1 > best_val_acc:
            best_val_acc = val_acc1
            stop_counter = 0
            checkpoint_name = f"/{epoch}_val_acc_{val_acc1:.3f}.pt"
            torch.save(model.state_dict(), checkpoint_folder + "/" + checkpoint_name)
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

    test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_acc10 = evaluate_test(
        cfg, class_encoding, test_loader, path, checkpoint_folder, checkpoint_name, device, criterion
    )
    run.log(
            {
                "test_loss": test_loss,
                "test_acc1": test_acc1,
                "test_acc2": test_acc2,
                "test_acc3": test_acc3,
                "test_acc4": test_acc4,
                "test_acc5": test_acc5,
                "test_acc10": test_acc10
        }
    )

    run.finish()

    return


if __name__ == "__main__":
    train()
