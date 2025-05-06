import argparse

def get_indices_from_task_id(list1, list2, list3, list4, list5, list6, list7, list8, task_id):
    """Generates unique indices for a given task ID based on the lengths of input lists."""
    indices = []
    base = 1
    lists = [list1, list2, list3, list4, list5, list6, list7, list8]

    total_combinations = 1
    for lst in lists:
        total_combinations *= len(lst)

    if task_id >= total_combinations:
        raise ValueError(f"task_id {task_id} is out of range for the given lists. Maximum allowed is {total_combinations - 1}.")

    for lst in lists:
        length = len(lst)
        indices.append((task_id // base) % length)
        base *= length

    return indices

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rates', nargs='+', type=float, help="Learning rates")
    parser.add_argument('-s', '--schedulers', nargs='+', type=str, help="Schedulers")
    parser.add_argument('-o', '--optimizers', nargs='+', type=str, help="Optimizers")
    parser.add_argument('-w', '--decays', nargs='+', type=float, help="Decays")
    parser.add_argument('-a', '--architectures', nargs='+', type=str, help="Architectures")
    parser.add_argument('-k', '--special_keys', nargs='+', type=str, help="Special Keys")
    parser.add_argument('-d', '--datasets', nargs='+', type=str, help="Datasets")
    parser.add_argument('-b', '--batch_size', nargs='+', type=int, help="Batch size")
    parser.add_argument('--id', type=int, help="SLURM_ARRAY_TASK_ID")
    args = parser.parse_args()

    learning_rates = args.learning_rates
    schedulers = args.schedulers
    optimizers = args.optimizers
    decays = args.decays
    architectures = args.architectures
    special_keys = args.special_keys
    datasets = args.datasets
    batch_size = args.batch_size

    task_id = args.id
    indices = get_indices_from_task_id(learning_rates, schedulers, optimizers, decays, architectures, special_keys, datasets, batch_size, task_id)

    params = [
        learning_rates[indices[0]],
        schedulers[indices[1]],
        optimizers[indices[2]],
        decays[indices[3]],
        architectures[indices[4]],
        special_keys[indices[5]],
        datasets[indices[6]],
        batch_size[indices[7]]
    ]

    print(' '.join(map(str, params)))

if __name__ == "__main__":
    main()
