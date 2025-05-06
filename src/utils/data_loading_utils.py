import torchaudio
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchaudio.transforms import TimeMasking, FrequencyMasking, MelSpectrogram, TimeStretch
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

WAVEFORM_MEAN_LIST = [-1.196521e-5]
WAVEFORM_STD_LIST = [0.029951086]
WAVEFORM_MEAN = -1.196521e-5
WAVEFORM_STD = 0.029951086


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
    # transforms.Normalize(mean=WAVEFORM_MEAN, std=WAVEFORM_MEAN),
    to_mel_spectrogram,
    pad_spectrogram,
    # transforms.Normalize(mean=WAVEFORM_MEAN_LIST, std=WAVEFORM_MEAN_LIST),
])


AUG_TRANSFORMS = transforms.Compose([
    # transforms.Normalize(mean=WAVEFORM_MEAN, std=WAVEFORM_MEAN),
    to_mel_spectrogram,
    RandomTimeStretch(p=0.8),
    pad_spectrogram,
    RandomFrequencyMasking(p=0.5),
    RandomTimeMasking(time_mask_param=3, p=0.5),
    RandomFrequencyMasking(freq_mask_param=1, p=0.6),
    RandomTimeMasking(time_mask_param=1, p=0.6),
    # transforms.Normalize(mean=WAVEFORM_MEAN_LIST, std=WAVEFORM_MEAN_LIST),
])


def normalize_waveform(waveform, mean=WAVEFORM_MEAN, std=WAVEFORM_MEAN):
    """Normalize waveform before converting to spectrogram."""
    return (waveform - mean) / std

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

class AudioDataset(Dataset):
    def __init__(self, root, dataset, transform=True, transform_aug=False, special_keys=False, class_idx=None):
        self.dataset = dataset
        self.root = root
        self.data_root = self.root if self.dataset=='all' else os.path.join(root, dataset)
        self.transform = TRANSFORMS if transform and not transform_aug else None
        self.transform_aug = AUG_TRANSFORMS if transform_aug else None
        self.all_classes = self.get_classes()
        self.unmapped_classes = self.all_classes[0] if not special_keys else self.all_classes[1]
        self.classes = self.all_classes[0] if not special_keys else self.all_classes[2]
        if class_idx is None:
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        else:
            self.class_to_idx = class_idx
            self.classes = class_idx.keys()

        self.file_paths = self._get_file_paths()

    def get_classes(self):
        special = ['apos', 'backslash', 'bracketclose', 'bracketopen', 'caps', 'comma', 'delete',
        'dot', 'down', 'enter', 'equal', 'esc', 'fn', 'lctrl', 'lcmd', 'lalt', 'left', 'lshift', 'ralt',
        'rctrl', 'rshift', 'right', 'semicolon', 'slash', 'space', 'start', 'tab', 'up']

        if self.dataset == 'all':
            all_classes = []
            for dataset in ['mka', 'practical', 'noiseless']:
                dataset_path = os.path.join(self.root, dataset)
                if os.path.isdir(dataset_path):
                    all_classes.extend(os.listdir(dataset_path))
            all_classes = list(set(all_classes))
            classes = [cls_name for cls_name in all_classes if cls_name not in special]
        else:
            classes = [cls_name for cls_name in os.listdir(self.data_root) if cls_name not in special]
            all_classes = os.listdir(self.data_root)

        mapped_classes = [MAP.get(key, key) for key in all_classes]

        return sorted(classes), sorted(all_classes), sorted(np.unique(mapped_classes))

    def _get_file_paths(self):
        paths = []
        datasets = [self.dataset] if self.dataset != 'all' else [ 'mka', 'practical', 'noiseless']

        for data in datasets:
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
        waveform = normalize_waveform(waveform)
        waveform = waveform.to(torch.float32)
        if self.transform:
            waveform = self.transform(waveform)
        if self.transform_aug:
            waveform = self.transform_aug(waveform)
        label = self.class_to_idx[cls_name]

        return waveform, label