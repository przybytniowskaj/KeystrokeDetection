import os
import torch
import torchaudio
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import detect_leading_silence

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


def isolator(signal, sample_rate, size, scan, before, after, threshold):
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
        if timestamp >= prev_end - 0.12 * sample_rate:
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


def plot_energy(signal, energy, threshold, borders, sample_rate, output_dir_img, key, dataset, subfolder, save_plots=False, show_plots=False):
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(signal)
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1)

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
    plt.figure(figsize=(12, 5))
    librosa.display.waveshow(energy)
    plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1)

    plt.title(f'Energy - {len(borders)} {key}-strokes found')
    if save_plots:
        plt.savefig(os.path.join(output_dir_img, f"{dataset}_{subfolder}_{key.lower()}_energy.png"))
    if show_plots:
        plt.show()
    plt.close()


def process_audio_files(audio_file, output_dir, output_dir_img, dataset, num_keystrokes=25, save_plots=False, show_plots=False):
    for root, dirs, files in os.walk(audio_file):
        subfolder = os.path.basename(root)
        for file in tqdm(files, desc=f"Processing files in {subfolder}"):
            if file.endswith('.wav'):
                loc = os.path.join(root, file)
                waveform, sample_rate = librosa.load(loc, sr=22000)
                strokes = []
                threshold = 0.03
                step = 0.006
                key = file.split('.')[0]
                num_try = 0
                while not len(strokes) == num_keystrokes and num_try < 300:
                    strokes, energy, borders = isolator(waveform[1*sample_rate:], sample_rate, 512, 124, 10000, 10000, threshold)
                    if len(strokes) < num_keystrokes:
                        threshold -= step
                    if len(strokes) > num_keystrokes:
                        threshold += step
                    if threshold <= 0:
                        print('-- not possible for: ', file)
                        break
                    step = step * 0.99
                    num_try += 1
                if save_plots or show_plots:
                    plot_energy(waveform[1*sample_rate:], energy, threshold, borders, sample_rate, output_dir_img, key, dataset, subfolder, save_plots, show_plots)
                with open(os.path.join("./Data/", f"{dataset}_tresholds.txt"), 'a') as f:
                    f.write(f"{threshold}\n")
                for i, stroke in enumerate(strokes):
                    if dataset == 'practical_dl':
                        keyboard_type = 'mac'
                        recording_type = 'zoom' if subfolder == 'zoom' else 'live'
                    # function has been tested for practical_dl dataset only
                    elif dataset == 'MKA':
                        if subfolder == 'Zoom' or subfolder == 'Messenger':
                            recording_type = 'zoom'
                            keyboard_type = 'unknown'
                        else:
                            keyboard_type = subfolder
                            recording_type = 'live'
                    keystroke_filename = f"{output_dir}{keyboard_type}_{recording_type}_{key.lower()}_{i}.wav"
                    torchaudio.save(keystroke_filename, stroke, sample_rate)


def get_audio_lengths(directory):
    max_duration = 0
    min_duration = float('inf')
    mean_duration = 0
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            file_path = os.path.join(directory, filename)
            waveform, sample_rate = torchaudio.load(file_path)

            duration = waveform.size(1) / sample_rate
            mean_duration += duration
            if duration > max_duration:
                max_duration = duration
            if duration < min_duration:
                min_duration = duration
            count += 1
    mean_duration /= count
    return max_duration, min_duration, mean_duration, count


def trim_silence_in_directory(data_dir, trimmed_data_dir, min_length=400):
    trim_leading_silence = lambda x: x[max(0, detect_leading_silence(x) - min_length):]
    trim_trailing_silence = lambda x: trim_leading_silence(x.reverse()).reverse()
    strip_silence = lambda x: trim_trailing_silence(trim_leading_silence(x))

    if not os.path.exists(trimmed_data_dir):
        os.makedirs(trimmed_data_dir)
    count = 0
    for root, _, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.wav'):
                input_file_path = os.path.join(root, filename)
                output_file_path = os.path.join(trimmed_data_dir, filename)

                sound = AudioSegment.from_file(input_file_path)
                length_before = len(sound)
                stripped_sound = strip_silence(sound)
                length_after = len(stripped_sound)
                if length_before != length_after:
                    count += 1
                stripped_sound.export(output_file_path, format='wav')
    return count
