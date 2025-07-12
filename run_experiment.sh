#!/bin/bash
#SBATCH --account zychowski-lab
#SBATCH --partition=short
#SBATCH --nodes 1
#SBATCH --cpus-per-task 2
#SBATCH --gpus-per-task 1
#SBATCH --time 23:59:00
#SBATCH --mem=80G
#SBATCH --ntasks=1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=julia.przybytniowska.stud@pw.edu.pl
#SBATCH --job-name=diff_data_exp
#SBATCH --output=exp_logs/initial_swin_smaller-%A_%a.log
#SBATCH --array=0-15


script_path=$(readlink -f "$0")
cat $script_path

source /etc/environment

source /mnt/evafs/groups/zychowski-lab/jprzybytniowska/miniconda3/etc/profile.d/conda.sh
conda activate thesis
echo $CONDA_DEFAULT_ENV


wandb online
export HYDRA_FULL_ERROR=1
OPTIMIZERS=(adam)
ARCHITECTURES=( default bigger) #big
DECAYS=(0.05) #0.001 0.05
LEARNING_RATES=( 0.0001 0.00005) # 0.0001 0.00001 0.00005
SCHEDULERS=( chained CosineAnnealingLR ) #chained
SPECIALKEYS=(false true) #false
DATASETS=(all ) #  custom_noisy all all_w_custom
LINEAR_INIT=(true) #  false
# EXCLUDE_FEW_SP=(true)

PARAMS=($(python configs/return_unique_param_set.py -l "${LEARNING_RATES[@]}" -s "${SCHEDULERS[@]}" -o "${OPTIMIZERS[@]}" -w "${DECAYS[@]}" -a "${ARCHITECTURES[@]}" -k "${SPECIALKEYS[@]}" -d "${DATASETS[@]}" -b "${LINEAR_INIT[@]}" --id $SLURM_ARRAY_TASK_ID | tr -d '[],'))

LR=${PARAMS[0]}
SCHEDULER=${PARAMS[1]}
OPTIMIZER=${PARAMS[2]}
DECAY=${PARAMS[3]}
ARCHITECTURE=${PARAMS[4]}
SPECIALKEYS=${PARAMS[5]}
DATASETS=${PARAMS[6]}
LINEAR_INIT=${PARAMS[7]}
# EXCLUDE_FEW_SP=${EXCLUDE_FEW_SP[0]}

echo "LR: $LR"
echo "SCHEDULER: $SCHEDULER"
echo "OPTIMIZER: $OPTIMIZER"
echo "DECAY: $DECAY"
echo "ARCHITECTURE: $ARCHITECTURE"
echo "SPECIALKEYS: $SPECIALKEYS"
echo "DATASETS: $DATASETS"
echo "LINEAR_INIT: $LINEAR_INIT"

srun python src/train_model.py \
    ++lr=$LR ++optimizer=$OPTIMIZER \
    ++model_params=$ARCHITECTURE \
    ++scheduler=$SCHEDULER \
    ++weight_decay=$DECAY \
    ++special_keys=$SPECIALKEYS \
    ++dataset=$DATASETS \
    ++init_linear=$LINEAR_INIT \
