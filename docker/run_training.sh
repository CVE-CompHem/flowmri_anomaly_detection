#!/bin/bash

set -euo pipefail

HPC_PREDICT_ANOMALY_DETECTION_IMAGE=${HPC_PREDICT_ANOMALY_DETECTION_IMAGE:-'cscs-ci/hpc-predict/anomaly_detection/deploy'}
HPC_PREDICT_DATA_DIR=$(realpath $1)
ANOMALY_DETECTION_TRAINING_DATA_DIR=$2 # such as 2021-05-31_11-46-39_copper

if [ "$#" -eq 3 ]; then
    time_stamp_host="$3"
else
    time_stamp_host=$(date +'%Y-%m-%d_%H-%M-%S')_$(hostname)
fi

# TODO: Possibly take into account input data shape (produced by flownet), adds another data dependency

relative_training_directory="segmenter/segmented_data/${ANOMALY_DETECTION_TRAINING_DATA_DIR}/training_data.hdf5"
host_training_directory="${HPC_PREDICT_DATA_DIR}/${relative_training_directory}"
container_training_directory="/hpc-predict-data/${relative_training_directory}"

echo "Host training data directory: ${host_training_directory}"

chmod -R u+w "${host_training_directory}"

#TODO: dataset name can be an argument to the script

relative_output_directory="anomaly_detection//hpc_predict/v1/training/${time_stamp_host}_flownet"
host_output_directory="${HPC_PREDICT_DATA_DIR}/${relative_output_directory}"
container_output_directory="/hpc-predict-data/${relative_output_directory}"

mkdir -p ${host_output_directory}
echo "Host output directory: \"${host_output_directory}\""

shell_command=$(printf "%s" \
    "source /src/hpc-predict/flowmri_anomaly_detection/venv/bin/activate && " \
    "PYTHONPATH=/src/hpc-predict/flowmri_anomaly_detection:/src/hpc-predict/hpc-predict-io/python python3 -u " \
/Users/hande     "/src/hpc-predict/flowmri_anomaly_detection/train.py " \
    "--training_input \"${container_training_directory}\" " \
    "--training_output \"${container_output_directory}\" ")

set -x
docker run --rm -u $(id -u ${USER}):$(id -g ${USER}) -v ${HPC_PREDICT_DATA_DIR}:/hpc-predict-data --entrypoint bash "${HPC_PREDICT_ANOMALY_DETECTION_IMAGE}" -c "${shell_command}"
#singularity exec --nv -B ${HPC_PREDICT_DATA_DIR}:/hpc-predict-data "/scratch-second/hpc-predict/anomaly_detection.img" bash -c '${shell_command}'
set +x

chmod -R a=rX "${host_training_directory}"
