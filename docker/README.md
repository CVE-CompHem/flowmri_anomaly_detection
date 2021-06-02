### Dockerized anomaly detection

Requires docker installed on host. If you followed the instructions on main page, ci/fetch_containers builds all necessary containers, including the anomaly detection container. To build the anomaly detection docker image locally, run 

```bash
./build_deploy.sh
./build_debug.sh <PYDEVD_PYCHARM_VERSION>
```

###

# Anomaly Detection 

Currently, Anomaly Detection works with SegmentedFlowMRI data, output of CNN Segmenter. All the documentation is based on using output of CNN Segmenter. When impact module is ready, anomaly detection should use output of impact. 

TODO: Change documentation so that anomaly detection uses output of impact. 

## Preprocessing 

Anomaly Detection Preprocessing is same with CNN Segmenter preprocessing. If you followed the instructions and got a `training_data.hdf5` in `data/v1/decrypt/segmenter/segmented_data/${time_stamp_host}`, you can skip to Training part. 

TODO: Change how to get random walker segmentations when they can be directly pulled from pollux with dvc. 

Anomaly Detection is trained based on random walker segmentations of the dataset. These random walker segmentations are done manually and uploaded to Pollux. Please download it from pollux's `shared_data` folder, save it to `data/v1/encrypt/segmenter/random_walker_segmenter` directory and extract it.

Anomaly Detection does preprocessing of data before training the model:

First, it copies fully sampled FlowMRI data, under sampled reconstructed FlowMRI data and random walker segmentations into a work directory called `data/v1/decrypt/segmenter/segmented_data/${time_stamp_host}`.

The following script is for reorganizing the data and it takes three argument: 
1- `time_stamp_host` of fully sampled FlowMRI data which is relative to `data/v1/decrypt/flownet/hpc_predict/v2/inference`
2- `time_stamp_host` of under sampled reconstructed FlowMRI data which is relative to `data/v1/decrypt/flownet/hpc_predict/v2/inference` 
3- `time_stamp_host` of random walker segmentations which is relative to `data/v1/decrypt/segmenter/random_walker_segmenter` 

```
./segmenter/cnn_segmenter_for_mri_4d_flow/data_flownet_reorganize_dir_structure.sh 2021-02-11_19-41-32_daint102 2021-03-19_15-46-05_daint102 2021-02-11_20-14-44_daint102

```
Secondly, these reorganized data in `segmented_data/${time_stamp_host}` directory are combined to a single hdf5 file, called `training_data.hdf5` . This output file is saved in same work directory.

The first argument corresponds to output directory where container's output should be written and second argument corresponds to segmenter training work directory, relative to `segmenter/segmented_data`.

```
./flowmri_anomaly_detection/docker/run_training_preprocessing.sh data/v1 2021-05-31_11-46-39_copper 
```
## Training

After preprocessing is complete and `training_data.hdf5` file is obtained, anomaly detection can be trained. The first argument corresponds to data directory, second corresponds to segmenter training work directory where `training_data.hdf5` is stored. This script trains the anomaly detection and saves the model under `data/v1/decrypt/anomaly_detection/hpc_predict/v1/training/${time_stamp_host}`

```
./flowmri_anomaly_detection/docker/run_training.sh data/v1 2021-05-31_11-46-39_copper
```

## Inference

To run inference, script needs the data directory, anomaly detection model, which is saved under `/data/v1/anomaly_detection/hpc_predict/v1/training/${time_stamp_host}` and segmenter inference file.

```
./flowmri_anomaly_detection/docker/run_inference.sh data/v1 2021-05-31_13-34-40_copper 2021-04-20_00-33-21_copper_volN5_R10/output/kspc_R10_volN5_vn.mat_segmented.h5

```


