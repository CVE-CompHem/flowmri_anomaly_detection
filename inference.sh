#!/bin/bash
#
# Script to send job to SLURM clusters using sbatch.
# Usage: sbatch main.sh
# Adjust line '-l hostname=xxxxx' before runing.
# The script also requires changing the paths of the CUDA and python environments and the code to the local equivalents of your machines.

## SLURM Variables:
#SBATCH  --output=logs/%j.out
#SBATCH  --gres=gpu:1
#SBATCH  --mem=30G
#SBATCH  --exclude='biwirender12','biwirender08','biwirender05','biwirender14'
#SBATCH  --priority='TOP'

# activate virtual environment
source /usr/bmicnas01/data-biwi-01/nkarani/softwares/anaconda/installation_dir/bin/activate tf_v1_15

## EXECUTION OF PYTHON CODE:
python /usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/code/code/hpc-predict/flowmri_anomaly_detection/inference.py \
--training_output '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/code/code/hpc-predict/flowmri_anomaly_detection/logdir/' \
--inference_input '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/eth_ibt/flownet/pollux/all_data/v4_seg_rw.h5' \
--inference_output '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/eth_ibt/flownet/pollux/all_data/v4_R6_anomaly.h5'

echo "Hostname was: `hostname`"
echo "Reached end of job file."