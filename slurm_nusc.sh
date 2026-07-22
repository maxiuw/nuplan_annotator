#!/usr/bin/env bash
#SBATCH -J nusc_caption
#SBATCH -A Berzelius-2026-200
#SBATCH -t 1-00:00:00
#SBATCH --gpus 1
#SBATCH --output /proj/berzelius-2023-364/users/x_macwo/code/nuplan_captioner/logs/%J_%a_nusc.out
#SBATCH --error  /proj/berzelius-2023-364/users/x_macwo/code/nuplan_captioner/logs/%J_%a_nusc.err
#SBATCH --array=0-7

echo "Starting job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}"

. /proj/berzelius-2023-364/users/x_macwo/mambaforge/etc/profile.d/conda.sh
conda activate dml_cuda

DATA_ROOT=/proj/berzelius-2023-364/users/x_macwo/code/nuplan_annot/nuplan_annotator/data_nusc
SCRIPT=/proj/berzelius-2023-364/users/x_macwo/code/nuplan_annot/nuplan_annotator/caption_scenes.py
OUTPUT_DIR=/proj/berzelius-2023-364/users/x_macwo/code/nuplan_captioner/outputs
MODEL=/proj/berzelius-2023-364/users/x_macwo/models/Qwen2-VL-7B-Instruct

mkdir -p ${OUTPUT_DIR} /proj/berzelius-2023-364/users/x_macwo/code/nuplan_captioner/logs

python ${SCRIPT} \
    --dataset      nuscenes \
    --data_root    ${DATA_ROOT} \
    --nusc_version v1.0-trainval \
    --camera       CAM_FRONT \
    --model        ${MODEL} \
    --output       ${OUTPUT_DIR}/nusc_shard${SLURM_ARRAY_TASK_ID}.jsonl \
    --stride       1 \
    --shard        ${SLURM_ARRAY_TASK_ID} \
    --num_shards   ${SLURM_ARRAY_TASK_COUNT}
