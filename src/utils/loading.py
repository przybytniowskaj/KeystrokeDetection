import os
import random
import numpy as np
import torch
import torchaudio
import noisereduce as nr
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchaudio.transforms import (
    TimeMasking,
    FrequencyMasking,
    MelSpectrogram,
    TimeStretch,
)
import torch.nn.functional as F
from utils.stats import load_waveform_stats, DATASET_GROUPS, EXCLUDED_KEYS
from constants.loading import TEST_DATASETS

import warnings

warnings.filterwarnings("ignore")


def normalize_waveform(waveform, dataset_name, special_keys=False):
    stats = load_waveform_stats(special_keys)
    if dataset_name not in stats:
        raise ValueError(f"No stats found for dataset key: {dataset_name}")

    mean = stats[dataset_name]["mean"]
    std = stats[dataset_name]["std"]
    return (waveform - mean) / std


class TimeShifting:
    def __call__(self, samples):
        samples = samples.numpy()
        shift = int(samples.shape[1] * 0.3)
        random_shift = random.randint(0, shift)
        data_roll = np.zeros_like(samples)
        data_roll[0] = np.roll(samples[0], random_shift)
        data_roll[1] = np.roll(samples[1], random_shift)
        return torch.tensor(data_roll)


def pad_spectrogram(spectrogram, target_size=64):
    """Pad or crop spectrogram to ensure [1, target_size, target_size] shape"""
    spectrogram = spectrogram.log2()
    _, mel_bins, time_steps = spectrogram.shape

    if mel_bins > target_size:
        spectrogram = spectrogram[:, :target_size, :]
    elif mel_bins < target_size:
        pad_mel = target_size - mel_bins
        spectrogram = F.pad(spectrogram, (0, 0, 0, pad_mel))

    if time_steps > target_size:
        spectrogram = spectrogram[:, :, :target_size]
    elif time_steps < target_size:
        pad_time = target_size - time_steps
        spectrogram = F.pad(spectrogram, (0, pad_time, 0, 0))
    return spectrogram


class RandomFrequencyMasking:
    def __init__(self, freq_mask_param=3, p=0.5):
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_mask_param)
        self.p = p

    def __call__(self, spectrogram):
        if torch.rand(1).item() < self.p:
            return self.freq_mask(spectrogram).to(torch.float32)
        return spectrogram


class RandomTimeMasking:
    def __init__(self, time_mask_param=3, p=0.5):
        self.time_mask = TimeMasking(time_mask_param=time_mask_param)
        self.p = p

    def __call__(self, spectrogram):
        if torch.rand(1).item() < self.p:
            return self.time_mask(spectrogram).to(torch.float32)
        return spectrogram


class RandomTimeStretch:
    def __init__(self, p=0.5, range_1=(0.8, 0.9), range_2=(1.2, 1.3), n_freq=64):
        self.rate_1 = torch.empty(1).uniform_(*range_1).item()
        self.rate_2 = torch.empty(1).uniform_(*range_2).item()
        self.time_stretch_1 = TimeStretch(fixed_rate=self.rate_1, n_freq=n_freq)
        self.time_stretch_2 = TimeStretch(fixed_rate=self.rate_2, n_freq=n_freq)
        self.p = p

    def __call__(self, spectrogram):
        rand_num = torch.rand(1).item()
        if rand_num < self.p * 0.5:
            return self.time_stretch_1(spectrogram).to(torch.float32)
        elif rand_num < self.p:
            return self.time_stretch_2(spectrogram).to(torch.float32)
        return spectrogram


class AudioDataset(Dataset):
    def __init__(
        self,
        root,
        dataset,
        transform_aug=False,
        special_keys=False,
        class_idx=None,
        image_size=64,
        exclude_few_special_keys=False,
        sample_rate=22000,
        noise_reduction=False,
    ):
        self.image_size = image_size
        self.sample_rate = sample_rate
        self.dataset = dataset
        self.root = root
        self.noise_reduction = noise_reduction
        self.data_folders = DATASET_GROUPS.get(self.dataset, [self.dataset])

        transformations_short = transforms.Compose(
            [
                MelSpectrogram(
                    self.sample_rate,
                    n_mels=self.image_size,
                    n_fft=512,
                    hop_length=512 // 4,
                ),
                lambda x: pad_spectrogram(x, target_size=self.image_size),
            ]
        )
        transformations_w_aug_short = transforms.Compose(
            [
                MelSpectrogram(
                    self.sample_rate,
                    n_mels=self.image_size,
                    n_fft=512,
                    hop_length=512 // 4,
                ),
                RandomTimeStretch(p=0.8, n_freq=self.image_size),
                lambda x: pad_spectrogram(x, target_size=self.image_size),
                RandomFrequencyMasking(p=0.4),
                RandomTimeMasking(time_mask_param=np.random.uniform(1, 4), p=0.4),
                RandomFrequencyMasking(freq_mask_param=np.random.uniform(1, 5), p=0.4),
                RandomTimeMasking(time_mask_param=np.random.uniform(1, 6), p=0.4),
            ]
        )
        self.transform_short = (
            transformations_w_aug_short if transform_aug else transformations_short
        )

        transformations_long = transforms.Compose(
            [
                MelSpectrogram(
                    self.sample_rate,
                    n_mels=self.image_size,
                    n_fft=1024,
                    hop_length=1024 // 4,
                ),
                lambda x: pad_spectrogram(x, target_size=self.image_size),
            ]
        )
        transformations_w_aug_long = transforms.Compose(
            [
                MelSpectrogram(
                    self.sample_rate,
                    n_mels=self.image_size,
                    n_fft=1024,
                    hop_length=1024 // 4,
                ),
                RandomTimeStretch(p=0.8, n_freq=self.image_size),
                lambda x: pad_spectrogram(x, target_size=self.image_size),
                RandomFrequencyMasking(p=0.4),
                RandomTimeMasking(time_mask_param=np.random.uniform(1, 4), p=0.4),
                RandomFrequencyMasking(freq_mask_param=np.random.uniform(1, 5), p=0.4),
                RandomTimeMasking(time_mask_param=np.random.uniform(1, 6), p=0.4),
            ]
        )
        self.transform_long = (
            transformations_w_aug_long if transform_aug else transformations_long
        )

        self.all_classes = self.get_classes(exclude_few_special_keys)
        self.classes = self.all_classes[0] if not special_keys else self.all_classes[1]
        # self.classes = self.all_classes[0] if not special_keys else self.all_classes[2]
        if class_idx is None:
            self.class_to_idx = {
                cls_name: idx for idx, cls_name in enumerate(self.classes)
            }
        else:
            self.class_to_idx = class_idx
            self.classes = class_idx.keys()

        self.file_paths = self._get_file_paths()

    def get_classes(self, exclude_few_special_keys):
        special = set(
            [
                "apos",
                "backslash",
                "bracketclose",
                "bracketopen",
                "caps",
                "comma",
                "delete",
                "fn",
                "start",
                "dot",
                "down",
                "enter",
                "equal",
                "esc",
                "lctrl",
                "lcmd",
                "lalt",
                "left",
                "lshift",
                "ralt",
                "rcmd",
                "rctrl",
                "rshift",
                "right",
                "semicolon",
                "slash",
                "space",
                "start",
                "tab",
                "up",
                "dash",
                "cmd",
                "ctrl",
            ]
        )
        excluded = EXCLUDED_KEYS if exclude_few_special_keys else set()

        all_classes = []
        for dataset in self.data_folders:
            dataset_path = os.path.join(self.root, dataset)
            if os.path.isdir(dataset_path):
                all_classes.extend(os.listdir(dataset_path))
        all_classes = list(set(all_classes))
        classes = [cls_name for cls_name in all_classes if cls_name not in special]

        all_classes = [cls_name for cls_name in all_classes if cls_name not in excluded]
        # mapped_classes = [MAP.get(key, key) for key in all_classes]

        return sorted(classes), sorted(
            all_classes
        )  # , sorted(np.unique(mapped_classes))

    def _get_file_paths(self):
        paths = []

        for data in self.data_folders:
            for cls_name in self.classes:  # to be fixed
                # mapped_classes = [k for k, v in MAP.items() if v == cls_name] + [cls_name]
                # for key in mapped_classes:
                folder = os.path.join(self.root, data, cls_name)
                if os.path.isdir(folder):
                    for file in os.listdir(os.path.join(folder)):
                        paths.append((os.path.join(folder, file), cls_name))
        return paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, cls_name = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        length = waveform.shape[1] / sr

        if self.noise_reduction:
            waveform = self.apply_noise_reduction(waveform, sr)

        waveform = normalize_waveform(waveform, self.dataset).to(torch.float32)
        if length > 0.5:
            waveform = self.transform_long(waveform)
        else:
            waveform = self.transform_short(waveform)

        label = self.class_to_idx[cls_name]

        return waveform, label

    def apply_noise_reduction(self, waveform, sr):
        waveform_denoised = nr.reduce_noise(y=waveform, sr=sr)
        return waveform_denoised


def get_all_dataloaders(cfg, ROOT_DIR, DATA_DIR):
    train_dataset = AudioDataset(
        ROOT_DIR + DATA_DIR + "/train",
        cfg.dataset,
        transform_aug=cfg.transform_aug,
        special_keys=cfg.special_keys,
        exclude_few_special_keys=cfg.exclude_few_special_keys,
        image_size=cfg.image_size,
    )
    num_classes = len(train_dataset.classes)
    class_encoding = train_dataset.class_to_idx

    val_dataset = AudioDataset(
        ROOT_DIR + DATA_DIR + "/val",
        cfg.dataset,
        transform_aug=False,
        special_keys=cfg.special_keys,
        exclude_few_special_keys=cfg.exclude_few_special_keys,
        image_size=cfg.image_size,
    )
    # batch_size = cfg.batch_size
    batch_size = len(train_dataset) // 10
    batch_size = 2 ** (batch_size.bit_length() - 2)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    test_loaders = {}
    for name in TEST_DATASETS:
        dataset = AudioDataset(
            ROOT_DIR + DATA_DIR + "/test",
            name,
            transform_aug=False,
            special_keys=cfg.special_keys,
            class_idx=train_dataset.class_to_idx,
            exclude_few_special_keys=cfg.exclude_few_special_keys,
            image_size=cfg.image_size,
        )
        test_loaders[name] = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_loaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loaders,
    }

    return all_loaders, num_classes, class_encoding, batch_size
