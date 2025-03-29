import torchaudio
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

WAVEFORM_MEAN = [-1.196521e-5]
WAVEFORM_STD = [0.029951086]


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
    _, mel_bins, time_steps = spectrogram.shape  # Should be [1, 64, T]

    if time_steps < target_size:
        pad_amount = target_size - time_steps
        spectrogram = F.pad(spectrogram, (0, pad_amount))  # Pad on time axis
    else:
        spectrogram = spectrogram[:, :, :target_size]  # Crop to target size
    return spectrogram


to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(25000, n_mels=64, n_fft=1024, hop_length=320)


TRANSFORMS = transforms.Compose([
    # transforms.Normalize(mean=WAVEFORM_MEAN, std=WAVEFORM_MEAN),
    to_mel_spectrogram,
    pad_spectrogram,
    transforms.Normalize(mean=WAVEFORM_MEAN, std=WAVEFORM_MEAN),
])


AUG_TRANSFORMS = transforms.Compose([
    # transforms.Normalize(mean=WAVEFORM_MEAN, std=WAVEFORM_MEAN),
    to_mel_spectrogram,
    pad_spectrogram,
    transforms.FrequencyMasking(3),
    transforms.TimeMasking(3),
    transforms.FrequencyMasking(2),
    transforms.TimeMasking(2),
    transforms.Normalize(mean=WAVEFORM_MEAN, std=WAVEFORM_MEAN),
])


def normalize_waveform(waveform, mean=WAVEFORM_MEAN, std=WAVEFORM_MEAN):
    """Normalize waveform before converting to spectrogram."""
    return (waveform - mean) / std


class AudioDataset(Dataset):
    def __init__(self, root, transform=True, transform_aug=False, train_mean=WAVEFORM_MEAN, train_std=WAVEFORM_MEAN):
        self.root = root
        self.transform = TRANSFORMS if transform else None
        self.transform_aug = AUG_TRANSFORMS if transform_aug else None
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.file_paths = self._get_file_paths()
        # self.train_mean = train_mean
        # self.train_std = train_std

    def _get_file_paths(self):
        paths = []
        for cls_name in self.classes:
            for file in os.listdir(os.path.join(self.root, cls_name)):
                paths.append(
                    (
                        os.path.join(self.root, cls_name, file),
                        cls_name,
                    )
                )
        return paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, cls_name = self.file_paths[idx]
        waveform, _ = torchaudio.load(file_path)
        # waveform = normalize_waveform(waveform, self.train_mean, self.train_std)
        if self.transform:
            waveform = self.transform(waveform)
        if self.transform_aug:
            waveform = self.transform_aug(waveform)
        label = self.class_to_idx[cls_name]

        return waveform, label