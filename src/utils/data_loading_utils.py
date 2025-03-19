import torchaudio
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchaudio.transforms import TimeMasking, FrequencyMasking, MelSpectrogram, TimeStretch
import torch.nn.functional as F

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
        print('timemask')
        if torch.rand(1).item() < self.p:
            return self.time_mask(spectrogram).to(torch.float32)
        return spectrogram


class RandomTimeStretch:
    def __init__(self, p=0.5, range_1=(0.8, 0.9), range_2=(1.2, 1.3)):
        self.rate_1 = torch.empty(1).uniform_(*range_1).item()
        print(self.rate_1)
        self.rate_2 = torch.empty(1).uniform_(*range_2).item()
        print(self.rate_2)
        self.time_stretch_1 = TimeStretch(fixed_rate=self.rate_1, n_freq=64)
        self.time_stretch_2 = TimeStretch(fixed_rate=self.rate_2, n_freq=64)
        self.p = p

    def __call__(self, spectrogram):
        rand_num = torch.rand(1).item()
        print(rand_num)
        print(self.rate_1)
        print(self.rate_2)
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
    RandomTimeStretch(p=0.4),
    pad_spectrogram,
    RandomFrequencyMasking(p=0.3),
    RandomTimeMasking(time_mask_param=3, p=0.3),
    RandomFrequencyMasking(freq_mask_param=1, p=0.4),
    RandomTimeMasking(time_mask_param=1, p=0.4),
    # transforms.Normalize(mean=WAVEFORM_MEAN_LIST, std=WAVEFORM_MEAN_LIST),
])


def normalize_waveform(waveform, mean=WAVEFORM_MEAN, std=WAVEFORM_MEAN):
    """Normalize waveform before converting to spectrogram."""
    return (waveform - mean) / std


class AudioDataset(Dataset):
    def __init__(self, root, transform=True, transform_aug=False):
        self.root = root
        # assert not ( transform and transform_aug ), 'choose only one tranformation'

        self.transform = TRANSFORMS if transform and not transform_aug else None
        self.transform_aug = AUG_TRANSFORMS if transform_aug else None
        self.classes = sorted(os.listdir(root))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.file_paths = self._get_file_paths()

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
        waveform = normalize_waveform(waveform)
        waveform = waveform.to(torch.float32)
        if self.transform:
            waveform = self.transform(waveform)
        if self.transform_aug:
            waveform = self.transform_aug(waveform)
        label = self.class_to_idx[cls_name]

        return waveform, label