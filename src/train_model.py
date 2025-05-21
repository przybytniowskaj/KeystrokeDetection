import torch
import pandas as pd
import json
import os
from tqdm import tqdm
import torchaudio
import random
import numpy as np
import wandb
import hydra
wandb.login()
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from utils.data_loading_utils import get_all_dataloaders
from utils.train_eval_utils import MODELS, train_epoch, evaluate_test, evaluate_model, init_linear, separate_parameters

OPTIMIZERS = {
    'adam': torch.optim.Adam,
    'adamw': torch.optim.AdamW,
    'sgd': torch.optim.SGD,
    'rmsprop': torch.optim.RMSprop,
}

SCHEDULERS = {
    'StepLR': torch.optim.lr_scheduler.StepLR,
    'CosineAnnealingLR': torch.optim.lr_scheduler.CosineAnnealingLR,
    'OneCycleLR': torch.optim.lr_scheduler.OneCycleLR,
    'CosineAnnealingWarmRestarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
}

ROOT_DIR = '/mnt/evafs/groups/zychowski-lab/jprzybytniowska/KeystrokeDetection'
DATA_DIR = '/data/final'

torchaudio.set_audio_backend('soundfile')

@hydra.main(config_path='../configs', config_name='config')
def train(cfg: DictConfig):
    os.chdir(ROOT_DIR)
    print(f'Current working directory: {os.getcwd()}')
    print(OmegaConf.to_yaml(cfg))

    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    data_loaders, num_classes, class_encoding = get_all_dataloaders(cfg, ROOT_DIR, DATA_DIR)
    train_loader = data_loaders['train']
    print(f'Train dataset size: {len(train_loader.dataset)}')
    val_loader = data_loaders['val']
    print(f'Validation dataset size: {len(val_loader.dataset)}')
    test_loader = data_loaders['test']

    model_params = cfg.model_configs[cfg.model][cfg.model_params]
    model = MODELS[cfg.model](num_classes=num_classes, **model_params)
    model.to(device)
    # model.apply(init_linear)
    model_name = cfg.model
    num_epochs = cfg.num_epochs
    criterion = torch.nn.CrossEntropyLoss()

    parameter_groups = []
    if cfg.partial_wd:
        param_dict = {pn: p for pn, p in model.named_parameters()}
        parameters_decay, parameters_no_decay = separate_parameters(model)
        parameter_groups = [
            {'params': [param_dict[pn] for pn in parameters_decay], 'weight_decay': cfg.weight_decay},
            {'params': [param_dict[pn] for pn in parameters_no_decay], 'weight_decay': 0.0},
        ]

    optimizer_params = cfg.optimizer_configs[cfg.optimizer]
    parameters = parameter_groups if cfg.partial_wd else model.parameters()
    optimizer = OPTIMIZERS[cfg.optimizer](
        parameters, **optimizer_params
    )

    if cfg.scheduler == 'chained':
        cosine_scheduler = SCHEDULERS['CosineAnnealingWarmRestarts'](
            optimizer, T_0=10, T_mult=1
        )
        poli_scheduler = torch.optim.lr_scheduler.PolynomialLR(
            optimizer, total_iters=cfg.num_epochs, power=0.9
        )
        scheduler = torch.optim.lr_scheduler.ChainedScheduler(
            [cosine_scheduler, poli_scheduler]
        )
    else:
        scheduler_params = cfg.scheduler_configs[cfg.scheduler]
        if cfg.scheduler == 'OneCycleLR':
            scheduler_params['max_lr'] = cfg.lr * 10
        scheduler = SCHEDULERS[cfg.scheduler](
            optimizer, **scheduler_params
        )

    str_wd = 'partial_wd' if cfg.partial_wd else 'wd'
    run_name = f'{model_name}_{cfg.model_params}_{cfg.optimizer}_{cfg.scheduler}_lr_{cfg.lr}_{str_wd}_{cfg.weight_decay}_special_keys_{cfg.special_keys}_{cfg.dataset}_{cfg.batch_size}'
    run = wandb.init(
        entity='przybytniowskaj-warsaw-university-of-technology',
        project=cfg.project_name,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        group='new_transformations',
    )

    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    path = f'{cfg.interim_results}{run_name}'
    os.makedirs(path, exist_ok=True)

    cfg.class_encoding = class_encoding

    with open(f'{path}/config.json', 'w') as f:
        json.dump(OmegaConf.to_container(cfg), f)

    columns = ['Train Loss', 'Train 1st Accuracy', 'Train 2nd Accuracy', 'Train 3rd Accuracy',
               'Train 4th Accuracy', 'Train 5th Accuracy', 'Train 10th Accuracy', 'Val Loss',
               'Val 1st Accuracy', 'Val 2nd Accuracy', 'Val 3rd Accuracy', 'Val 4th Accuracy',
               'Val 5th Accuracy', 'Val 10th Accuracy']
    df = pd.DataFrame(columns=columns)

    stop_counter_loss = 0
    stop_counter_acc = 0
    max_no_improvement = cfg.max_no_improvement
    best_val_loss = float('inf')
    best_val_acc = 0

    for epoch in (
            epoch_bar := tqdm(range(num_epochs), leave=False, desc='epoch', position=0)
        ):
        loss, acc1, acc2, acc3, acc4, acc5, acc10 = train_epoch(
            device, model, criterion, optimizer, train_loader
        )
        run.log(
            {
                'train_loss': loss,
                'train_acc1': acc1,
                'train_acc2': acc2,
                'train_acc3': acc3,
                'train_acc4': acc4,
                'train_acc5': acc5,
                'train_acc10': acc10
            }
        )
        val_loss, val_acc1, val_acc2, val_acc3, val_acc4, val_acc5, val_acc10 = evaluate_model(
            device, model, criterion, val_loader
        )
        run.log(
            {
                'val_loss': val_loss,
                'val_acc1': val_acc1,
                'val_acc2': val_acc2,
                'val_acc3': val_acc3,
                'val_acc4': val_acc4,
                'val_acc5': val_acc5,
                'val_acc10': val_acc10
            }
        )
        if epoch == 0:
            checkpoint_folder = path + '/model_checkpoints'
            os.makedirs(checkpoint_folder, exist_ok=True)
            checkpoint_name = f'/{epoch}_val_acc_{val_acc1:.3f}.pt'
            torch.save(model.state_dict(), checkpoint_folder + '/' + checkpoint_name)
        if val_acc1 > best_val_acc:
            for filename in os.listdir(checkpoint_folder):
                file_path = os.path.join(checkpoint_folder, filename)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            checkpoint_name = f'/{epoch}_val_acc_{round(val_acc1*100)}.pt'
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
                    'train acc': f'{acc1:.3f}',
                    'train loss': f'{loss:.3f}',
                    'val acc': f'{val_acc1:.3f}',
                    'val loss': f'{val_loss:.3f}',
                    'best val loss': f'{best_val_loss:.3f}',
                    'no imp loss': stop_counter_loss,
                    'no imp acc': stop_counter_acc
                }
            )
        if stop_counter_loss > max_no_improvement or stop_counter_acc > max_no_improvement:
            print('Early stopping, no improvement in validation accuracy or loss')
            break

        if cfg.scheduler == 'ReduceLROnPlateau':
            scheduler.step(metrics=val_loss)
        else:
            scheduler.step()

    # Logging train and validation loss and accuracies to wandb
    my_table = wandb.Table(dataframe=df)
    run.log({'table_key': my_table})

    # Calculating test accuracy and confusion matrixes and logging to wandb
    for name, loader in test_loader.items():
        test_loss, test_acc1, test_acc2, test_acc3, test_acc4, test_acc5, test_acc10 = evaluate_test(
            model_name, model_params, class_encoding, loader,
            path, checkpoint_folder, checkpoint_name, device, criterion, name
        )

        run.log({
            f'test_loss_{name}': test_loss,
            f'test_acc1_{name}': test_acc1,
            f'test_acc2_{name}': test_acc2,
            f'test_acc3_{name}': test_acc3,
            f'test_acc4_{name}': test_acc4,
            f'test_acc5_{name}': test_acc5,
            f'test_acc10_{name}': test_acc10,
        })

        cm_path = f'{path}/confusion_matrix_{name}.png'
        try:
            image = Image.open(cm_path)
            run.log({f'confusion_matrix_{name}': wandb.Image(image, caption=f'Confusion Matrix: {name}')})
        except FileNotFoundError:
            print(f'Confusion matrix not found for {name}, skipping.')

    run.finish()

    return


if __name__ == '__main__':
    train()
