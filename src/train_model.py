import torch
import pandas as pd
import json
import os
import argparse
from tqdm import tqdm
import torchaudio
import random
import numpy as np
import wandb
import hydra
wandb.login()
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from utils.data_loading_utils import AudioDataset
from utils.train_eval_utils import MODELS, train_epoch, evaluate_test, evaluate_model, init_linear, separate_parameters

OPTIMIZERS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
}

SCHEDULERS = {
    "StepLR": torch.optim.lr_scheduler.StepLR,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
    "OneCycleLR": torch.optim.lr_scheduler.OneCycleLR,
    "CosineAnnealingWarmRestarts": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
}

ROOT_DIR = '/mnt/evafs/groups/zychowski-lab/jprzybytniowska/KeystrokeDetection'
DATA_DIR = '/data/final'

torchaudio.set_audio_backend("soundfile")

@hydra.main(config_path="../configs", config_name="config")
def train(cfg: DictConfig):
    os.chdir(ROOT_DIR)
    print(f"Current working directory: {os.getcwd()}")
    print(OmegaConf.to_yaml(cfg))

    torch.cuda.empty_cache()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    train_dataset = AudioDataset(
        ROOT_DIR + DATA_DIR + '/train', cfg.dataset, transform_aug=cfg.transform_aug, transform=not(cfg.transform_aug),
        special_keys=cfg.special_keys
    )
    val_dataset = AudioDataset(
        ROOT_DIR + DATA_DIR + '/val', cfg.dataset, transform_aug=False, special_keys=cfg.special_keys
    )
    test_dataset_all = AudioDataset(
        ROOT_DIR + DATA_DIR + '/test', 'all', transform_aug=False, special_keys=cfg.special_keys, class_idx=train_dataset.class_to_idx
    )
    test_dataset_1 = AudioDataset(
        ROOT_DIR + DATA_DIR + '/test', 'practical', transform_aug=False, special_keys=cfg.special_keys, class_idx=train_dataset.class_to_idx
    )

    test_dataset_2 = AudioDataset(
        ROOT_DIR + DATA_DIR + '/test', 'noiseless', transform_aug=False, special_keys=cfg.special_keys, class_idx=train_dataset.class_to_idx
    )

    test_dataset_3 = AudioDataset(
        ROOT_DIR + DATA_DIR + '/test', 'mka', transform_aug=False, special_keys=cfg.special_keys, class_idx=train_dataset.class_to_idx
    )

    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    class_encoding = train_dataset.class_to_idx

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader_all = torch.utils.data.DataLoader(
        test_dataset_all,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader_1 = torch.utils.data.DataLoader(
        test_dataset_1,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader_2 = torch.utils.data.DataLoader(
        test_dataset_2,
        batch_size=cfg.batch_size,
        shuffle=False,
    )
    test_loader_3 = torch.utils.data.DataLoader(
        test_dataset_3,
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model_params = cfg.model_configs[cfg.model][cfg.model_params]
    model = MODELS[cfg.model](num_classes=num_classes, **model_params)
    model.to(device)
    model.apply(init_linear)
    model_name = cfg.model
    num_epochs = cfg.num_epochs
    criterion = torch.nn.CrossEntropyLoss()

    # param_dict = {pn: p for pn, p in model.named_parameters()}
    # parameters_decay, parameters_no_decay = separate_parameters(model)
    # parameter_groups = [
    #     {"params": [param_dict[pn] for pn in parameters_decay], "weight_decay": cfg.weight_decay},
    #     {"params": [param_dict[pn] for pn in parameters_no_decay], "weight_decay": 0.0},
    # ]

    optimizer_params = cfg.optimizer_configs[cfg.optimizer]
    # parameters = parameter_groups if cfg.partial_wd else model.parameters()
    parameters = model.parameters()
    optimizer = OPTIMIZERS[cfg.optimizer](
        parameters, **optimizer_params
    )

    scheduler_params = cfg.scheduler_configs[cfg.scheduler]
    if cfg.scheduler == "OneCycleLR":
        scheduler_params["max_lr"] = cfg.lr * 10
    scheduler = SCHEDULERS[cfg.scheduler](
        optimizer, **scheduler_params
    )

    str_wd = "partial_wd" if cfg.partial_wd else 'wd'
    run_name = f"{model_name}_{cfg.model_params}_{cfg.optimizer}_{cfg.scheduler}_lr_{cfg.lr}_{str_wd}_{cfg.weight_decay}_special_keys_{cfg.special_keys}_{cfg.dataset}_{cfg.batch_size}_correct"
    run = wandb.init(
        entity="przybytniowskaj-warsaw-university-of-technology",
        project=cfg.project_name,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    path = f"{cfg.interim_results}{run_name}"
    os.makedirs(path, exist_ok=True)

    with open(f"{path}/config.json", "w") as f:
        json.dump(OmegaConf.to_container(cfg), f)

    columns = ["Train Loss", "Train 1st Accuracy", "Train 2nd Accuracy", "Train 3rd Accuracy",
               "Train 4th Accuracy", "Train 5th Accuracy", "Train 10th Accuracy", "Val Loss",
               "Val 1st Accuracy", "Val 2nd Accuracy", "Val 3rd Accuracy", "Val 4th Accuracy",
               "Val 5th Accuracy", "Val 10th Accuracy"]
    df = pd.DataFrame(columns=columns)

    stop_counter_loss = 0
    stop_counter_acc = 0
    max_no_improvement = cfg.max_no_improvement
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in (
            epoch_bar := tqdm(range(num_epochs), leave=False, desc="epoch", position=0)
        ):
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
            for filename in os.listdir(checkpoint_folder):
                file_path = os.path.join(checkpoint_folder, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            checkpoint_name = f"/{epoch}_val_acc_{round(val_acc1*100)}.pt"
            torch.save(model.state_dict(), checkpoint_folder + checkpoint_name)
            best_val_acc = val_acc1
            stop_counter_acc = 0
        else:
            stop_counter_acc += 1
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            stop_counter_loss = 0
        else:
            stop_counter_loss += 1

        df.loc[epoch] = [loss, acc1, acc2, acc3, acc4, acc5, acc10,
                         val_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5, val_acc10]

        epoch_bar.set_postfix(
                {
                    "train acc": f'{acc1:.3f}',
                    "train loss": f'{loss:.3f}',
                    "val acc": f'{val_acc1:.3f}',
                    "val loss": f'{val_loss:.3f}',
                    "best val loss": f'{best_val_loss:.3f}',
                    "no imp loss": stop_counter_loss,
                    "no imp acc": stop_counter_acc
                }
            )
        if stop_counter_loss > max_no_improvement or stop_counter_acc > max_no_improvement:
            print("Early stopping, no improvement in validation accuracy or loss")
            break

        if cfg.scheduler == "ReduceLROnPlateau":
            scheduler.step(metrics=val_loss)
        else:
            scheduler.step()


    my_table = wandb.Table(dataframe=df)
    run.log({"table_key": my_table})

    test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_acc10 = evaluate_test(
        model_name, model_params, class_encoding, test_loader_all, path, checkpoint_folder, checkpoint_name, device, criterion, 'all'
    )
    run.log(
            {
                "test_loss_all": test_loss,
                "test_acc1_all": test_acc1,
                "test_acc2_all": test_acc2,
                "test_acc3_all": test_acc3,
                "test_acc4_all": test_acc4,
                "test_acc5_all": test_acc5,
                "test_acc10_all": test_acc10
            }
    )

    test_loss_1, test_acc1_1, test_acc2_1, test_acc3_1, test_acc4_1, test_acc5_1, test_acc10_1 = evaluate_test(
        model_name, model_params, class_encoding, test_loader_1, path, checkpoint_folder, checkpoint_name, device, criterion, 'practical'
    )

    run.log(
            {
                "test_loss_practical": test_loss_1,
                "test_acc1_practical": test_acc1_1,
                "test_acc2_practical": test_acc2_1,
                "test_acc3_practical": test_acc3_1,
                "test_acc4_practical": test_acc4_1,
                "test_acc5_practical": test_acc5_1,
                "test_acc10_practical": test_acc10_1,
            }
    )

    test_loss_2, test_acc1_2, test_acc2_2, test_acc3_2, test_acc4_2, test_acc5_2, test_acc10_2 = evaluate_test(
        model_name, model_params, class_encoding, test_loader_2, path, checkpoint_folder, checkpoint_name, device, criterion, 'noiseless'
    )

    run.log(
            {
                "test_loss_noiseless": test_loss_2,
                "test_acc1_noiseless": test_acc1_2,
                "test_acc2_noiseless": test_acc2_2,
                "test_acc3_noiseless": test_acc3_2,
                "test_acc4_noiseless": test_acc4_2,
                "test_acc5_noiseless": test_acc5_2,
                "test_acc10_noiseless": test_acc10_2,
            }
    )

    test_loss_3, test_acc1_3, test_acc2_3, test_acc3_3, test_acc4_3, test_acc5_3, test_acc10_3 = evaluate_test(
        model_name, model_params, class_encoding, test_loader_3, path, checkpoint_folder, checkpoint_name, device, criterion, 'mka'
    )

    run.log(
            {
                "test_loss_mka": test_loss_3,
                "test_acc1_mka": test_acc1_3,
                "test_acc2_mka": test_acc2_3,
                "test_acc3_mka": test_acc3_3,
                "test_acc4_mka": test_acc4_3,
                "test_acc5_mka": test_acc5_3,
                "test_acc10_mka": test_acc10_3,
        }
    )

    image_all = Image.open(f"{path}/confusion_matrix_all.png")
    wandb_image = wandb.Image(image_all, caption="Sample Image")
    run.log({"confusion_matrix_all": wandb_image})
    image_1 = Image.open(f"{path}/confusion_matrix_practical.png")
    wandb_image = wandb.Image(image_1, caption="Sample Image")
    run.log({"confusion_matrix_practical": wandb_image})
    image_2 = Image.open(f"{path}/confusion_matrix_noiseless.png")
    wandb_image = wandb.Image(image_2, caption="Sample Image")
    run.log({"confusion_matrix_noiseless": wandb_image})
    image_3 = Image.open(f"{path}/confusion_matrix_mka.png")
    wandb_image = wandb.Image(image_3, caption="Sample Image")
    run.log({"confusion_matrix_mka": wandb_image})

    run.finish()

    return


if __name__ == "__main__":
    train()
