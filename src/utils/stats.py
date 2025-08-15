import os
import csv
import torchaudio
import torch
from tqdm import tqdm
from constants.loading import DATASET_GROUPS, ALPHANUMERIC_KEYS, EXCLUDED_KEYS


def load_waveform_stats(special_keys=False):
    path = (
        "data/final/waveform_stats_all.csv"
        if special_keys
        else "data/final/waveform_stats_alnum.csv"
    )
    stats = {}
    with open(path, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            stats[row["dataset"]] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }
    return stats


def compute_stats(dataset_root, DATASET_GROUPS, filter_fn=None):
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
            if filter_fn and not filter_fn(label):
                continue

            for fname in tqdm(os.listdir(label_path), desc=f"{folder}/{label}"):
                file_path = os.path.join(label_path, fname)
                waveform, _ = torchaudio.load(file_path)
                waveform = waveform.to(torch.float32)
                total_sum += waveform.sum().item()
                total_squared_sum += (waveform**2).sum().item()
                total_count += waveform.numel()

        if total_count > 0:
            mean = total_sum / total_count
            std = (total_squared_sum / total_count - mean**2) ** 0.5
            dataset_means_stds[folder] = {"mean": mean, "std": std}

    group_stats = {}
    for group_name, datasets in DATASET_GROUPS.items():
        group_means = [
            dataset_means_stds[d]["mean"] for d in datasets if d in dataset_means_stds
        ]
        group_stds = [
            dataset_means_stds[d]["std"] for d in datasets if d in dataset_means_stds
        ]
        if group_means:
            group_stats[group_name] = {
                "mean": sum(group_means) / len(group_means),
                "std": sum(group_stds) / len(group_stds),
            }

    return dataset_means_stds, group_stats


def write_stats_to_csv(path, dataset_stats, group_stats):
    with open(path, mode="w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["dataset", "mean", "std"])
        writer.writeheader()
        for dataset, stats in dataset_stats.items():
            writer.writerow(
                {"dataset": dataset, "mean": stats["mean"], "std": stats["std"]}
            )
        for group_name, stats in group_stats.items():
            writer.writerow(
                {"dataset": group_name, "mean": stats["mean"], "std": stats["std"]}
            )


def compute_and_save_all_stats(
    dataset_root, output_csv_path_all, output_csv_path_alnum
):
    all_stats, all_group_stats = compute_stats(
        dataset_root, DATASET_GROUPS, filter_fn=lambda key: key not in EXCLUDED_KEYS
    )
    write_stats_to_csv(output_csv_path_all, all_stats, all_group_stats)

    alnum_stats, alnum_group_stats = compute_stats(
        dataset_root, DATASET_GROUPS, filter_fn=lambda key: key in ALPHANUMERIC_KEYS
    )
    write_stats_to_csv(output_csv_path_alnum, alnum_stats, alnum_group_stats)

    return all_stats, all_group_stats, alnum_stats, alnum_group_stats
