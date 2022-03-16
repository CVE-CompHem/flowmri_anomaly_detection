#!/bin/bash

set -euo pipefail

source "$(dirname "$0")/../venv/bin/activate"

cd "$(dirname "$0")/../../hpc-predict-io/data_interfaces/"

set -x
python3 validate_data_interface.py --app ../../flowmri_anomaly_detection/data_interface/flowmri_anomaly_detection_data_interface.yml:inference
