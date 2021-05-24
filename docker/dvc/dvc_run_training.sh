#!/bin/bash

set -euo pipefail

HPC_PREDICT_ANOMALY_DETECTION_IMAGE=${HPC_PREDICT_ANOMALY_DETECTION_IMAGE:-'cscs-ci/hpc-predict/anomaly_detection/deploy'}
HPC_PREDICT_DATA_DIR=$(realpath $1)
TRAINING_INPUT=$(realpath $2)
TRAINING_OUTPUt=$(realpath $3)

if [ "$#" -eq 4 ]; then
    time_stamp_host="$4"
else
    time_stamp_host=$(date +'%Y-%m-%d_%H-%M-%S')_$(hostname)
fi

# TODO: Possibly take into account input data shape (produced by flownet), adds another data dependency

time_stamp_host_training="2020-04-30_18-36-47_neerav"
relative_input_directory="segmenter/random_walker_segmenter/hpc_predict/v1/${time_stamp_host_training}"
host_input_directory="${HPC_PREDICT_DATA_DIR}/${relative_input_directory}"
container_input_directory="/hpc-predict-data/${relative_input_directory}"

echo "Host training data directory: ${host_input_directory}"

chmod -R u+w "${host_input_directory}"

relative_output_directory="segmenter/cnn_segmenter/hpc_predict/v1/training/${time_stamp_host}"
host_output_directory="${HPC_PREDICT_DATA_DIR}/${relative_output_directory}"
container_output_directory="/hpc-predict-data/${relative_output_directory}"

# FIXME: Training data (--training-input) and trained model (--training-output)
#  storage location held in JSON-file in config directory,
#  especially training input parameter currently is an unused parameter
shell_command=$(printf "%s" \
  " \"$(realpath --relative-to="${host_output_directory}" $(realpath $(dirname $0)))/dvc_check_code_consistency.sh\" \"${HPC_PREDICT_SEGMENTER_IMAGE}\" && " \
  " docker run --rm -u $(id -u ${USER}):$(id -g ${USER}) -v ${HPC_PREDICT_DATA_DIR}:/hpc-predict-data \"${HPC_PREDICT_SEGMENTER_IMAGE}\" " \
  " /src/hpc-predict/segmenter/docker/dvc/cnn_training.sh \"${container_input_directory}\" \"${container_output_directory}/output\" ")

set -x
mkdir -p "${host_output_directory}/output"
cd "${host_output_directory}"
dvc run --no-exec -n "cnn_segmenter_training_${time_stamp_host}" -d "${host_input_directory}" -o "${host_output_directory}/output" "${shell_command}"
#dvc run -n "cnn_segmenter_training_${time_stamp_host}" -d "${host_input_directory}" -o "${host_output_directory}/output" $(realpath --relative-to="${host_output_directory}" $(realpath $(dirname $0)))/dvc_check_code_consistency.sh "${HPC_PREDICT_SEGMENTER_IMAGE}" && docker run --rm -u $(id -u ${USER}):$(id -g ${USER}) -v ${HPC_PREDICT_DATA_DIR}:/hpc-predict-data "${HPC_PREDICT_SEGMENTER_IMAGE}" /src/hpc-predict/segmenter/docker/dvc/cnn_training.sh "${container_input_directory}" "${container_output_directory}/output"
#dvc run -n "cnn_segmenter_training_${time_stamp_host}" -d "${host_input_directory}" -o "${host_output_directory}/output" docker run --rm -u $(id -u ${USER}):$(id -g ${USER}) -v ${HPC_PREDICT_DATA_DIR}:/hpc-predict-data "${HPC_PREDICT_SEGMENTER_IMAGE}" /src/hpc-predict/segmenter/docker/dvc/cnn_training.sh "${container_input_directory}" "${container_output_directory}/output"
cd -
set +x
