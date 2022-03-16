#!/bin/bash

#set -euo pipefail

set -x

HPC_PREDICT_DATA_DIR=$(realpath $1)
ANOMALY_DETECTION_TRAINING_DIR=$2 # e.g. 2021-05-31_13-34-40_copper_flownet
SEGMENTER_INFERENCE_INPUT=$3 # e.g. 2021-01-28_22-45-08_daint102_volN7_R16/output/kspc_R16_volN7_vn.mat_segmented.h5 (output directory of cnn segmenter)

#TODO: Change time_stamp_host so that it indicates data belongs to which voluteer/patient under which undersampling ratio such as 
if [ "$#" -eq 4 ]; then
    time_stamp_host="$4"
else
    time_stamp_host=$(date +'%Y-%m-%d_%H-%M-%S')_$(hostname)
fi

# find main repository directory (this works only if this is a submodule of the main repository)
MAIN_REPO_DIR="$(dirname $(realpath $0))"
while [[ ! -x ${MAIN_REPO_DIR}/ci/fetch_containers.sh && ${MAIN_REPO_DIR} != "/" ]] ; do
    MAIN_REPO_DIR=$(dirname "${MAIN_REPO_DIR}")
done
if [[ ! -x "${MAIN_REPO_DIR}"/data/container_scripts/run_docker_or_sarus.sh ]] ; then
    echo "Could not find the script container_scripts/run_docker_or_sarus.sh. Is this a separate clone, and not a full clone of the main repository?"
    exit 1
fi

# Add decrypt prefix in case of a data directory with encryption
if [ -d "${HPC_PREDICT_DATA_DIR}/encrypt" ]; then
  echo "Identified ${HPC_PREDICT_DATA_DIR} as directory with encryption."
  pgrep_encfs=$(pgrep --list-full encfs)
#  if echo "${pgrep_encfs}" | grep -q "${HPC_PREDICT_DATA_DIR}/decrypt"; then
    echo "encfs is running on ${HPC_PREDICT_DATA_DIR}/decrypt."
    encrypt_dir="encrypt/"
    decrypt_dir="decrypt/"
    dvc_dir="config/"
#  else
#    echo "encfs seems not to be running - must be launched first."
#    exit 1
#  fi
else
  echo "Identified ${HPC_PREDICT_DATA_DIR} as directory without encryption."
  encrypt_dir=""
  decrypt_dir=""
  dvc_dir=""
fi

relative_model_directory="anomaly_detection/hpc_predict/v1/training/${ANOMALY_DETECTION_TRAINING_DIR}"
host_encrypt_model_directory="${HPC_PREDICT_DATA_DIR}/${encrypt_dir}${relative_model_directory}"
host_decrypt_model_directory="${HPC_PREDICT_DATA_DIR}/${decrypt_dir}${relative_model_directory}"
container_model_directory="/hpc-predict-data/${relative_model_directory}"

if [ ! -d ${host_encrypt_model_directory} ]; then
    echo "Anomaly detection model directory \"${host_encrypt_model_directory}\" with trained model for inference does not exist. Exiting..."
    exit 1
fi

relative_input_file="segmenter/cnn_segmenter/hpc_predict/v1/inference/${SEGMENTER_INFERENCE_DIR}"
host_encrypt_input_file="${HPC_PREDICT_DATA_DIR}/${encrypt_dir}${relative_input_file}"
host_decrypt_input_file="${HPC_PREDICT_DATA_DIR}/${decrypt_dir}${relative_input_file}"
container_input_file="/hpc-predict-data/${relative_input_file}"

if [ ! -f ${host_encrypt_input_file} ]; then
    echo "Anomaly detection input directory \"${host_encrypt_input_file}\" with input data for inference does not exist. Exiting..."
    exit 1
fi

relative_output_directory="anomaly_detection/hpc_predict/v1/inference/${time_stamp_host}"
host_encrypt_output_directory="${HPC_PREDICT_DATA_DIR}/${encrypt_dir}${relative_output_directory}"
host_decrypt_output_directory="${HPC_PREDICT_DATA_DIR}/${decrypt_dir}${relative_output_directory}"
container_output_directory="/hpc-predict-data/${relative_output_directory}/output"

host_dvc_directory="${HPC_PREDICT_DATA_DIR}/${dvc_dir}${relative_output_directory}"

# TODO: declare config as dependency

mkdir -p "${host_dvc_directory}"
MAIN_REPO_RELATIVE=$(realpath --relative-to="${host_dvc_directory}" "${MAIN_REPO_DIR}")
echo ${MAIN_REPO_RELATIVE}
DATA_DIR_RELATIVE=$(realpath --relative-to="${host_dvc_directory}" "${HPC_PREDICT_DATA_DIR}")

shell_command_container=$(printf "%s" \
    "source /src/hpc-predict/flowmri_anomaly_detection/venv/bin/activate && " \
    "set -x && " \
    "mkdir -p \"${container_output_directory}\" && " \
    "cd /src/hpc-predict/flowmri_anomaly_detection && " \
    "LD_LIBRARY_PATH=/usr/local/cuda/lib64 " \
    "PYTHONPATH=/src/hpc-predict/flowmri_anomaly_detection python3 " \
        "inference.py " \
        "--training_output \"${container_model_directory}/\" " \
        "--inference_input \"${container_input_file}\" " \
	"--inference_output \"${container_output_directory}/$(echo $(basename ${container_input_file}) | sed -e 's/_segmented.h5/_segmented_anomaly.h5/')\" ")

# For docker, use the following shell_command
shell_command=$(printf "%s" \
   " ${MAIN_REPO_RELATIVE}/data/container_scripts/run_docker_or_sarus.sh docker run --rm -u \$(id -u \${USER}):\$(id -g \${USER}) --gpus all -v \$(pwd)/${DATA_DIR_RELATIVE}/${decrypt_dir}:/hpc-predict-data --entrypoint bash \"\${HPC_PREDICT_ANOMALY_DETECTION_IMAGE:-cscs-ci/hpc-predict/anomaly_detection/deploy}\" " \
   " -c '${shell_command_container}'")

# For singularity, use the following shell_command. Adjust your hpc-predict-data directory and anomaly_detection.img path accordingly.

#shell_command=$(printf "%s" \
#	" singularity exec --nv -B "/scratch-second/hpc-predict/data/v1/decrypt:/hpc-predict-data" "/scratch-second/hpc-predict/anomaly_detection.img"  bash -c '${shell_command_container}' ")

set -x

mkdir -p "${host_encrypt_output_directory}/output"
cd "${host_dvc_directory}"
dvc run --no-exec -n "anomaly_detection_inference_${time_stamp_host}" -d "${host_encrypt_model_directory}" -d "${host_encrypt_input_file}" -o "${host_encrypt_output_directory}/output" "${shell_command}"
set +x
