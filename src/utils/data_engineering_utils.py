import os
import random
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import detect_leading_silence
from src.utils.segmentation_constants import LABEL_MAP, DATASET_CONFIG

random.seed(42)

warnings.filterwarnings("ignore")


def get_middle_peaks(peaks, max_gap=5):
    if len(peaks) == 0:
        return []

    grouped_peaks = []
    current_group = [peaks[0]]

    for i in range(1, len(peaks)):
        if peaks[i] - peaks[i - 1] <= max_gap:
            current_group.append(peaks[i])
        else:
            grouped_peaks.append(current_group)
            current_group = [peaks[i]]

    grouped_peaks.append(current_group)
    refined_peaks = [group[len(group) // 2] for group in grouped_peaks]
    return refined_peaks


def isolator(signal, sample_rate, size, scan, before, after, threshold, overlap_rate):
    strokes = []
    stroke_boarders = []
    fft = librosa.stft(signal, n_fft=size, hop_length=scan)
    energy = np.abs(np.sum(fft, axis=0)).astype(float)
    energy = (energy - np.min(energy)) / (np.max(energy) - np.min(energy))

    threshed = energy > threshold
    peaks = np.where(threshed == True)[0]
    refined_peaks = get_middle_peaks(peaks)

    peak_count = len(refined_peaks)
    prev_end = sample_rate * 0.1 * (-1)
    for i in range(peak_count):
        this_peak = refined_peaks[i]
        timestamp = (this_peak * scan) + size // 2
        if timestamp >= prev_end - overlap_rate * sample_rate:
            if timestamp - before < 0:
                temp_before = 0
            else:
                temp_before = timestamp - before
            if timestamp + after > len(signal):
                temp_after = len(signal)
            else:
                temp_after = timestamp + after

            keystroke = signal[temp_before:temp_after]

            strokes.append(torch.tensor(keystroke)[None, :])
            stroke_boarders.append((temp_before, temp_after))
            prev_end = timestamp + after
    return strokes, energy, stroke_boarders


def plot_energy(signal, energy, threshold, borders, sample_rate, output_dir_img, key, dataset, subfolder,

                save_plots=False, show_plots=False):
    # signal plot
    # signal plot
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(signal)
    for idx, (start, end) in enumerate(borders):
        plt.axvline(x=start / sample_rate, color='black', linestyle='--', linewidth=1)
        plt.axvline(x=end / sample_rate, color='darkgrey', linestyle='--', linewidth=1)
        plt.text(start / sample_rate, threshold, f'{idx}', color='black', fontsize=6, va='bottom')
        plt.text(end / sample_rate, threshold, f'{idx}', color='black', fontsize=6, va='bottom')

    plt.title(f'Signal - {len(borders)} {key}-strokes found')
    if save_plots:
        plt.savefig(os.path.join(output_dir_img, f"{dataset}_{subfolder}_{key.lower()}.png"))
    if show_plots:
        plt.show()
    plt.close()

    # energy plot

    # energy plot
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(energy)
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1)
    plt.title(f'Energy - {len(borders)} {key}-strokes found')
    if save_plots:
        plt.savefig(os.path.join(output_dir_img, f"{dataset}_{subfolder}_{key.lower()}_energy.png"))
    if show_plots:
        plt.show()
    plt.close()


def split_data(data, splits=(0.7, 0.15, 0.15)):
    random.shuffle(data)
    total = len(data)
    train_end = int(total * splits[0])
    val_end = train_end + int(total * splits[1])

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def trim_silence_from_signal(signal, file_path, min_length=400):
    trim_leading_silence = lambda x: x[max(0, detect_leading_silence(x) - min_length):]
    trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
    strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))

    stripped_signal = strip_silence(signal)
    return np.frombuffer(stripped_signal.raw_data, dtype=np.int16)


def save_segmented_strokes(strokes, label, output_dir, dataset, subfolder, sample_rate, threshold, min_length):
    os.makedirs(os.path.join(output_dir, "tresholds"), exist_ok=True)
    with open(os.path.join(output_dir, "tresholds", f"{dataset}_tresholds.txt"), 'a') as f:
        f.write(f"{threshold}\n")

    keyboard_type = 'mac'
    recording_type = 'zoom' if dataset == 'practical' and subfolder == 'Zoom' else 'live'

    train_dir = os.path.join(output_dir, "train", dataset, label.lower())
    val_dir = os.path.join(output_dir, "val", dataset, label.lower())
    test_dir = os.path.join(output_dir, "test", dataset, label.lower())
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    filename_base = f"{keyboard_type}_{recording_type}_{label.lower()}"

    skip = None
    last_skip = None
    if 'keystrokes_to_skip' in DATASET_CONFIG[dataset] and label in DATASET_CONFIG[dataset]['keystrokes_to_skip']:
        skip = DATASET_CONFIG[dataset]['keystrokes_to_skip'][label]
    if 'last_strokes_to_skip' in DATASET_CONFIG[dataset] and label in DATASET_CONFIG[dataset]['last_strokes_to_skip']:
        last_skip = DATASET_CONFIG[dataset]['last_strokes_to_skip'][label]

    filtered_strokes = [
        (i, stroke) for i, stroke in enumerate(strokes)
        if not ((skip is not None and i in skip) or (last_skip is not None and i == len(strokes) - 1))
    ]

    train_strokes, val_strokes, test_strokes = split_data(filtered_strokes)

    for split_strokes, split_dir in [
        (train_strokes, train_dir),
        (val_strokes, val_dir),
        (test_strokes, test_dir),
    ]:
        for i, stroke in split_strokes:
            stroke_int16 = (stroke.numpy().flatten() * 32767).astype(np.int16)

            signal = AudioSegment(
                stroke_int16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
            )
            signal = trim_silence_from_signal(signal, f"{filename_base}_{i}.wav", min_length)
            filename = os.path.join(split_dir, f"{filename_base}_{i}.wav")
            torchaudio.save(
                filename,
                torch.tensor(signal, dtype=torch.int16).unsqueeze(0),
                sample_rate
            )

def split_data(data, splits=(0.7, 0.15, 0.15)):
    random.shuffle(data)
    total = len(data)
    train_end = int(total * splits[0])
    val_end = train_end + int(total * splits[1])

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data


def trim_silence_from_signal(signal, file_path, min_length=400):
    trim_leading_silence = lambda x: x[max(0, detect_leading_silence(x) - min_length):]
    trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
    strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))

    stripped_signal = strip_silence(signal)
    return np.frombuffer(stripped_signal.raw_data, dtype=np.int16)


def save_segmented_strokes(strokes, label, output_dir, dataset, subfolder, sample_rate, threshold, min_length):
    os.makedirs(os.path.join(output_dir, "tresholds"), exist_ok=True)
    with open(os.path.join(output_dir, "tresholds", f"{dataset}_tresholds.txt"), 'a') as f:
        f.write(f"{threshold}\n")

    keyboard_type = 'mac'
    recording_type = 'zoom' if dataset == 'practical' and subfolder == 'Zoom' else 'live'

    train_dir = os.path.join(output_dir, dataset, "train", label.lower())
    val_dir = os.path.join(output_dir, dataset, "val", label.lower())
    test_dir = os.path.join(output_dir, dataset, "test", label.lower())
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    filename_base = f"{keyboard_type}_{recording_type}_{label.lower()}"

    skip = None
    last_skip = None
    if 'keystrokes_to_skip' in DATASET_CONFIG[dataset] and label in DATASET_CONFIG[dataset]['keystrokes_to_skip']:
        skip = DATASET_CONFIG[dataset]['keystrokes_to_skip'][label]
    if 'last_strokes_to_skip' in DATASET_CONFIG[dataset] and label in DATASET_CONFIG[dataset]['last_strokes_to_skip']:
        last_skip = DATASET_CONFIG[dataset]['last_strokes_to_skip'][label]

    filtered_strokes = [
        (i, stroke) for i, stroke in enumerate(strokes)
        if not ((skip is not None and i in skip) or (last_skip is not None and i == len(strokes) - 1))
    ]

    train_strokes, val_strokes, test_strokes = split_data(filtered_strokes)

    for split_strokes, split_dir in [
        (train_strokes, train_dir),
        (val_strokes, val_dir),
        (test_strokes, test_dir),
    ]:
        for i, stroke in split_strokes:
            stroke_int16 = (stroke.numpy().flatten() * 32767).astype(np.int16)

            signal = AudioSegment(
                stroke_int16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
            )
            signal = trim_silence_from_signal(signal, f"{filename_base}_{i}.wav", min_length)
            filename = os.path.join(split_dir, f"{filename_base}_{i}.wav")
            torchaudio.save(
                filename,
                torch.tensor(signal, dtype=torch.int16).unsqueeze(0),
                sample_rate
            )


def process_audio_files(audio_file, output_dir, output_dir_img, dataset, save_plots=False,
                        show_plots=False):
    num_keystrokes = DATASET_CONFIG[dataset]['num_keystrokes']
    num_tries = DATASET_CONFIG[dataset]['num_segmentation_tries']
    before = DATASET_CONFIG[dataset]['before']
    after = DATASET_CONFIG[dataset]['after']
    steps = DATASET_CONFIG[dataset]['steps'] if 'steps' in DATASET_CONFIG[dataset] else None
    thresholds = DATASET_CONFIG[dataset]['thresholds'] if 'thresholds' in DATASET_CONFIG[dataset] else None
def process_audio_files(audio_file, output_dir, output_dir_img, dataset, save_plots=False,
                        show_plots=False):
    num_keystrokes = DATASET_CONFIG[dataset]['num_keystrokes']
    num_tries = DATASET_CONFIG[dataset]['num_segmentation_tries']
    before = DATASET_CONFIG[dataset]['before']
    after = DATASET_CONFIG[dataset]['after']
    steps = DATASET_CONFIG[dataset]['steps'] if 'steps' in DATASET_CONFIG[dataset] else None
    thresholds = DATASET_CONFIG[dataset]['thresholds'] if 'thresholds' in DATASET_CONFIG[dataset] else None
    for root, dirs, files in os.walk(audio_file):
        subfolder = os.path.basename(root)
        for file in tqdm(files, desc=f"Processing files in {subfolder}"):
            if file.endswith('.wav') or file.endswith('.m4a'):
                loc = os.path.join(root, file)
                waveform, sample_rate = librosa.load(loc, sr=22000)
                strokes = []
                key = os.path.splitext(file)[0].lower()
                key = LABEL_MAP.get(key, key)
                num_try = 0
                step = DATASET_CONFIG[dataset]['initial_step']
                threshold = DATASET_CONFIG[dataset]['initial_threshold']
                if steps is not None and key in DATASET_CONFIG[dataset]['steps']:
                    step = steps[key]
                if thresholds is not None and key in DATASET_CONFIG[dataset]['thresholds']:
                    threshold = thresholds[key]
                overlap_rate = DATASET_CONFIG[dataset]['overlap_rate']
                if 'overlap_rates' in DATASET_CONFIG[dataset] and key in DATASET_CONFIG[dataset]['overlap_rates']:
                    overlap_rate = DATASET_CONFIG[dataset]['overlap_rates'][key]

                while not len(strokes) == num_keystrokes and num_try < num_tries:
                    strokes, energy, borders = isolator(
                        waveform[1*sample_rate:], sample_rate, 512, 124, before, after, threshold, overlap_rate
                    )
                    if len(strokes) < num_keystrokes:
                        threshold -= step
                    if len(strokes) > num_keystrokes:
                        threshold += step
                    if threshold <= 0:
                        print('-- not possible for: ', file)
                        print('-- found: ', len(strokes))
                        break
                    step = step * 0.99
                    num_try += 1
                if save_plots or show_plots:
                    plot_energy(
                        waveform[1*sample_rate:], energy, threshold, borders, sample_rate,
                        output_dir_img, key, dataset, subfolder, save_plots, show_plots
                    )
                save_segmented_strokes(
                    strokes, key, output_dir, dataset, subfolder, sample_rate, threshold,
                    DATASET_CONFIG[dataset]['min_stroke_length']
                )
