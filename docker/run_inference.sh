#!/bin/bash

set -euo pipefail

# In this file, the outputs of segmenter are used as inference output. In case impact is involved, outputs of impact should be used as inference output. 

# Docker configuration
# HPC_PREDICT_ANOMALY_DETECTION_IMAGE=${HPC_PREDICT_ANOMALY_DETECTION_IMAGE:-'cscs-ci/hpc-predict/anomaly_detection/deploy'}

# Singularity configuration
HPC_PREDICT_ANOMALY_DETECTION_IMAGE=${HPC_PREDICT_ANOMALY_DETECTION_IMAGE:-'anomaly_detection.img'}
HPC_PREDICT_DATA_DIR=$(realpath $1)
ANOMALY_DETECTION_TRAINING_DIR=$2
SEGMENTER_INFERENCE_OUTPUT=$3

if [ "$#" -eq 4 ]; then
    time_stamp_host="$4"
else
    time_stamp_host=$(date +'%Y-%m-%d_%H-%M-%S')_$(hostname)
fi

SEGMENTER_INFERENCE_RESULT=$(basename ${SEGMENTER_INFERENCE_OUTPUT})
SEGMENTER_INFERENCE_DIR=$(dirname ${SEGMENTER_INFERENCE_OUTPUT})

relative_model_directory=anomaly_detection/hpc_predict/v1/training/${ANOMALY_DETECTION_TRAINING_DIR}
host_model_directory="${HPC_PREDICT_DATA_DIR}/${relative_model_directory}"
container_model_directory="/hpc-predict-data/${relative_model_directory}"

if [ -f ${host_model_directory} ]; then
    echo "Anomaly Detection model directory \"${host_model_directory}\" with trained model for inference does not exist. Exiting..."
    exit 1
fi

relative_input_directory=segmenter/cnn_segmenter/hpc_predict/v1/inference/${SEGMENTER_INFERENCE_DIR}
host_input_directory="${HPC_PREDICT_DATA_DIR}/${relative_input_directory}"
container_input_directory="/hpc-predict-data/${relative_input_directory}"

if [ -f ${host_input_directory} ]; then
    echo "Segmenter inference directory \"${host_input_directory}\" with input data for inference does not exist. Exiting..."
    exit 1
fi

relative_output_directory="flowmri_anomaly_detection/hpc_predict/v1/inference/${time_stamp_host}"
host_output_directory="${HPC_PREDICT_DATA_DIR}/${relative_output_directory}"
container_output_directory="/hpc-predict-data/${relative_output_directory}"

mkdir -p ${host_output_directory}
echo "Host output directory: \"${host_output_directory}\""

shell_command=$(printf "%s" \
    "source /src/hpc-predict/flowmri_anomaly_detection/venv/bin/activate && " \
    "set -x && " \
    "PYTHONPATH=/src/hpc-predict/flowmri_anomaly_detection:/src/hpc-predict/hpc-predict-io/python python3 -u " \
    "/src/hpc-predict/flowmri_anomaly_detection/inference.py" \
    "--training_output \"${container_model_directory}\" " \
    "--inference_input \"${container_input_directory}/${SEGMENTER_INFERENCE_RESULT}\" " \
    "--inference_output \"${container_output_directory}/segmented_anomaly.h5\" ")

set -x

# For docker, use the following
# docker run --rm -u $(id -u ${USER}):$(id -g ${USER}) -v ${HPC_PREDICT_DATA_DIR}:/hpc-predict-data --entrypoint bash "${HPC_PREDICT_ANOMALY_DETECTION_IMAGE}" -c "${shell_command}"

# For singularity, use the following
singularity exec --nv -B "/scratch/hharputlu/hpc-predict/data/v1/decrypt:/hpc-predict-data" "/scratch/hharputlu/hpc-predict/anomaly_detection.img" bash -c '${shell_command}'
set +x

