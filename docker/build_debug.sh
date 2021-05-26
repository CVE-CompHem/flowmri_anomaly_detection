#!/bin/bash

echo "Building container to contact Pycharm debug server (get pip package version 'PYDEVD_PYCHARM_VERSION' in Pycharm from Run > Edit configurations > Python Remote Debug/Debug Server)"

set -euo pipefail

cd "$(dirname "$0")"
COMMIT_SHA=$(git -C ../.. rev-parse HEAD)

if [ -z "${DEBUG_IMAGE-}" ]; then
    CONTAINER_REGISTRY=${CONTAINER_REGISTRY:-local}
fi

export ANOMALY_DETECTION_DEPLOY=${ANOMALY_DETECTION_DEPLOY:-"${CONTAINER_REGISTRY}/hpc-predict/anomaly_detection/deploy:${COMMIT_SHA}"}
export DEBUG_IMAGE=${DEBUG_IMAGE:-"${CONTAINER_REGISTRY}/hpc-predict/anomaly_detection/debug:${COMMIT_SHA}"}
export DEBUG_IMAGE_UNTAGGED=${DEBUG_IMAGE_UNTAGGED:-"${CONTAINER_REGISTRY}/hpc-predict/anomaly_detection/debug"}

MPI_MASTER_HOST=$(hostname)
if [[ -z $(echo ${MPI_MASTER_HOST} | grep -i daint) ]]; then
    set -x
    if [[ -z $(docker images -q ${ANOMALY_DETECTION_DEPLOY}) ]]; then
        echo "Error: Docker base image ${ANOMALY_DETECTION_DEPLOY} for anomaly detection not available"
        exit 1
    fi
    if [[ -z $(docker images -q ${DEBUG_IMAGE}) ]]; then
        docker build --build-arg ANOMALY_DETECTION_DEPLOY=${ANOMALY_DETECTION_DEPLOY} --build-arg PYDEVD_PYCHARM_VERSION=$1 -f debug/Dockerfile -t ${DEBUG_IMAGE} ..
        if [[ -n ${DEBUG_IMAGE_UNTAGGED} ]]; then
          docker tag ${DEBUG_IMAGE} ${DEBUG_IMAGE_UNTAGGED}
        fi
    fi
    set +x
fi

echo "Run \"export HPC_PREDICT_ANOMALY_DETECTION_IMAGE=${DEBUG_IMAGE_UNTAGGED}\" or \"export HPC_PREDICT_ANOMALY_DETECTION_IMAGE=${DEBUG_IMAGE}\" to automatically use the built image with docker/run scripts."
cd -
