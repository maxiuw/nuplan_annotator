#!/bin/bash
#SBATCH --job-name=nuplan_caption
#SBATCH --account=berzelius-2023-364
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/caption_%A_%a.out
#SBATCH --array=0-3          # 4 shards for mini (64 logs)

DATA_ROOT=/proj/berzelius-2023-364/users/x_macwo/code/nuplan_annot/data
SCRIPT_DIR=/proj/berzelius-2023-364/users/x_macwo/code/nuplan_captioner
OUTPUT_DIR=${SCRIPT_DIR}/outputs

mkdir -p ${OUTPUT_DIR} logs

module load Anaconda/2023.09-0-hpc1-bdist
conda activate nuplan   # adjust to your env name

python ${SCRIPT_DIR}/caption_scenes.py \
    --data_root   ${DATA_ROOT} \
    --split_dir   splits/mini \
    --blob_subdir sensor_blobs_mini \
    --output      ${OUTPUT_DIR}/captions_shard${SLURM_ARRAY_TASK_ID}.jsonl \
    --model       Qwen/Qwen2-VL-7B-Instruct \
    --camera      CAM_F0 \
    --stride      10 \
    --shard       ${SLURM_ARRAY_TASK_ID} \
    --num_shards  ${SLURM_ARRAY_TASK_COUNT}
