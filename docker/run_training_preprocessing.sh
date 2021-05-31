#!/bin/bash

set -euo pipefail

# This file is same with segmenter/docker/run_cnn_training_preprocessing.sh except they use different containers. Both segmenter and anomaly detection use same training_data.hdf5 file for training the model which is the output of this file.

HPC_PREDICT_ANOMALY_DETECTION_IMAGE=${HPC_PREDICT_ANOMALY_DETECTION_IMAGE:-'cscs-ci/hpc-predict/anomaly_detection/deploy'}
HPC_PREDICT_DATA_DIR=$(realpath $1)
HPC_PREDICT_TRAINING_WORK_DIR=$2 #relative with segmenter/segmented_data/ such as 2021-05-31_11-46-39_copper

relative_work_directory="segmenter/segmenter_data/${HPC_PREDICT_TRAINING_WORK_DIR}"
host_work_directory="${HPC_PREDICT_DATA_DIR}/${relative_work_directory}"
container_work_directory="/hpc-predict-data/${relative_work_directory}"

shell_command=$(printf "%s" \
    "source /src/hpc-predict/segmenter/random_walker_segmenter_for_mri_4d_flow/venv/bin/activate && " \
    "PYTHONPATH=/src/hpc-predict/segmenter/random_walker_segmenter_for_mri_4d_flow:/src/hpc-predict/hpc-predict-io/python python3 " \
    "/src/hpc-predict/segmenter/cnn_segmenter_for_mri_4d_flow/data_flownet_prepare_training_data.py \"${container_work_directory}\" ")

set -x
docker run --rm -u $(id -u ${USER}):$(id -g ${USER}) -v ${HPC_PREDICT_DATA_DIR}:/hpc-predict-data --entrypoint bash "${HPC_PREDICT_ANOMALY_DETECTION_IMAGE}" -c "${shell_command}"
#singularity exec --nv -B ${HPC_PREDICT_DATA_DIR}:/hpc-predict-data "/scratch-second/hpc-predict/anomaly_detection.img" bash -c '${shell_command}'
set +x

