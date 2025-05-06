import torchaudio
import os
import json
import random
import numpy as np
import torch
import csv
from collections import defaultdict
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import transforms
from torchaudio.transforms import TimeMasking, FrequencyMasking, MelSpectrogram, TimeStretch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")


def load_waveform_stats():
    with open("waveform_stats.json", "r") as f:
        return json.load(f)

WAVEFORM_STATS = load_waveform_stats()

def normalize_waveform(waveform, dataset_name):
    if dataset_name not in WAVEFORM_STATS:
        raise ValueError(f"No stats found for dataset key: {dataset_name}")

    mean = WAVEFORM_STATS[dataset_name]["mean"]
    std = WAVEFORM_STATS[dataset_name]["std"]
    return (waveform - mean) / std


class TimeShifting():
    def __call__(self, samples):
        samples = samples.numpy()
        shift = int(samples.shape[1] * 0.3)
        random_shift = random.randint(0, shift)
        data_roll = np.zeros_like(samples)
        data_roll[0] = np.roll(samples[0], random_shift)
        data_roll[1] = np.roll(samples[1], random_shift)
        return torch.tensor(data_roll)


def pad_spectrogram(spectrogram, target_size=64):
    """Pad or crop spectrogram to ensure [1, 64, 64] shape"""
    spectrogram = spectrogram.log2()
    _, mel_bins, time_steps = spectrogram.shape

    if time_steps < target_size:
        pad_amount = target_size - time_steps
        spectrogram = F.pad(spectrogram, (0, pad_amount))
    else:
        spectrogram = spectrogram[:, :, :target_size]
    return spectrogram


to_mel_spectrogram = MelSpectrogram(25000, n_mels=64, n_fft=1024, hop_length=320)


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
    def __init__(self, p=0.5, range_1=(0.8, 0.9), range_2=(1.2, 1.3)):
        self.rate_1 = torch.empty(1).uniform_(*range_1).item()
        self.rate_2 = torch.empty(1).uniform_(*range_2).item()
        self.time_stretch_1 = TimeStretch(fixed_rate=self.rate_1, n_freq=64)
        self.time_stretch_2 = TimeStretch(fixed_rate=self.rate_2, n_freq=64)
        self.p = p

    def __call__(self, spectrogram):
        rand_num = torch.rand(1).item()
        if rand_num < self.p * 0.5:
            return self.time_stretch_1(spectrogram).to(torch.float32)
        elif rand_num < self.p:
            return self.time_stretch_2(spectrogram).to(torch.float32)
        return spectrogram


TRANSFORMS = transforms.Compose([
    to_mel_spectrogram,
    pad_spectrogram,
])


AUG_TRANSFORMS = transforms.Compose([
    to_mel_spectrogram,
    RandomTimeStretch(p=0.8),
    pad_spectrogram,
    RandomFrequencyMasking(p=0.5),
    RandomTimeMasking(time_mask_param=3, p=0.5),
    RandomFrequencyMasking(freq_mask_param=1, p=0.6),
    RandomTimeMasking(time_mask_param=1, p=0.6),
])

MAP = {
            'lctrl': 'ctrl',
            'lcmd': 'cmd',
            'lalt': 'alt',
            'lshift': 'shift',
            'ralt': 'alt',
            'rctrl': 'ctrl',
            'rshift': 'shift',
            'rcmd': 'cmd',
            'bracketclose': 'bracket',
            'bracketopen': 'bracket'
        }

DATASET_GROUPS = {
    "all_w_custom": ['mka', 'practical', 'noiseless', 'custom_mac'],
    "custom_w_first_2": ['practical', 'noiseless', 'custom_mac'],
    "custom_w_first_2_w_noisy": ['practical', 'noiseless', 'custom_mac', 'custom_dishwasher', 'custom_open_window', 'custom_washing_machine'],
    "all_w_custom_noisy": ['mka', 'practical', 'noiseless', 'custom_mac', 'custom_dishwasher', 'custom_open_window', 'custom_washing_machine'],
    "all": ['mka', 'practical', 'noiseless'],
    "custom_noisy": ['custom_dishwasher', 'custom_open_window', 'custom_washing_machine'],
}


def compute_stats_all_datasets(dataset_root, output_csv_path):
    dataset_means_stds = {}

    for folder in os.listdir(dataset_root):
        folder_path = os.path.join(dataset_root, folder)
        if not os.path.isdir(folder_path):
            continue

        total_sum = 0.0
        total_squared_sum = 0.0
        total_count = 0

        for label in os.listdir(folder_path):
            label_path = os.path.join(folder_path, label)
            if not os.path.isdir(label_path):
                continue

            for fname in tqdm(os.listdir(label_path), desc=f"{folder}/{label}"):
                file_path = os.path.join(label_path, fname)
                waveform, _ = torchaudio.load(file_path)
                waveform = waveform.to(torch.float32)
                total_sum += waveform.sum().item()
                total_squared_sum += (waveform ** 2).sum().item()
                total_count += waveform.numel()

        if total_count > 0:
            mean = total_sum / total_count
            std = (total_squared_sum / total_count - mean ** 2) ** 0.5
            dataset_means_stds[folder] = {"mean": mean, "std": std}

    # Compute group means and stds
    group_stats = {}
    for group_name, datasets in DATASET_GROUPS.items():
        group_means = [dataset_means_stds[d]["mean"] for d in datasets if d in dataset_means_stds]
        group_stds = [dataset_means_stds[d]["std"] for d in datasets if d in dataset_means_stds]
        if group_means:
            group_stats[group_name] = {
                "mean": sum(group_means) / len(group_means),
                "std": sum(group_stds) / len(group_stds)
            }

    # Write all stats to CSV
    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["dataset", "mean", "std"])
        writer.writeheader()

        # Write per-dataset stats
        for dataset, stats in dataset_means_stds.items():
            writer.writerow({"dataset": dataset, "mean": stats["mean"], "std": stats["std"]})

        # Write group stats
        for group_name, stats in group_stats.items():
            writer.writerow({"dataset": group_name, "mean": stats["mean"], "std": stats["std"]})

    return dataset_means_stds, group_stats


class AudioDataset(Dataset):
    def __init__(self, root, dataset, transform=True, transform_aug=False, special_keys=False, class_idx=None,
                 exclude_few_special_keys=False):
        self.dataset = dataset
        self.root = root
        self.data_folders = DATASET_GROUPS.get(self.dataset, [self.dataset])
        self.transform = TRANSFORMS if transform and not transform_aug else None
        self.transform_aug = AUG_TRANSFORMS if transform_aug else None
        self.all_classes = self.get_classes(exclude_few_special_keys)
        self.unmapped_classes = self.all_classes[0] if not special_keys else self.all_classes[1]
        self.classes = self.all_classes[0] if not special_keys else self.all_classes[2]
        if class_idx is None:
            if special_keys:
                self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
            else:
                self.class_to_idx = {MAP.get(cls_name, cls_name): idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_idx
            self.classes = class_idx.keys()

        self.file_paths = self._get_file_paths()

    def get_classes(self, exclude_few_special_keys):
        special = ['apos', 'backslash', 'bracketclose', 'bracketopen', 'caps', 'comma', 'delete',
        'dot', 'down', 'enter', 'equal', 'esc', 'lctrl', 'lcmd', 'lalt', 'left', 'lshift', 'ralt',
        'rctrl', 'rshift', 'right', 'semicolon', 'slash', 'space', 'start', 'tab', 'up', 'dash', 'cmd', 'ctrl']
        excluded = ['fn', 'start'] if exclude_few_special_keys else []

        all_classes = []
        for dataset in self.data_folders:
            dataset_path = os.path.join(self.root, dataset)
            if os.path.isdir(dataset_path):
                all_classes.extend(os.listdir(dataset_path))
        all_classes = list(set(all_classes))
        classes = [cls_name for cls_name in all_classes if cls_name not in special]

        all_classes = [cls_name for cls_name in all_classes if cls_name not in excluded]
        mapped_classes = [MAP.get(key, key) for key in all_classes]

        return sorted(classes), sorted(all_classes), sorted(np.unique(mapped_classes))

    def _get_file_paths(self):
        paths = []

        for data in self.data_folders:
            for cls_name in self.classes:
                folder = os.path.join(self.root, data, cls_name)
                if os.path.isdir(folder):
                    for file in os.listdir(os.path.join(folder)):
                        paths.append(
                            (
                                os.path.join(folder, file),
                                MAP.get(cls_name, cls_name)
                            )
                        )
        return paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, cls_name = self.file_paths[idx]
        waveform, _ = torchaudio.load(file_path)
        waveform = normalize_waveform(waveform, self.dataset)
        waveform = waveform.to(torch.float32)
        if self.transform:
            waveform = self.transform(waveform)
        if self.transform_aug:
            waveform = self.transform_aug(waveform)
        label = self.class_to_idx[cls_name]

        return waveform, label
