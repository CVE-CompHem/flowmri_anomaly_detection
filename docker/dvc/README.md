Please make sure that you installed docker and dvc, docker containers are built correctly and dvc directories exist.

# Anomaly Detection

Currently, Anomaly Detection works with SegmentedFlowMRI data, output of CNN Segmenter. All the documentation is based on using output of CNN Segmenter. When impact module is ready, anomaly detection should use output of impact. 

TODO: Change documentation so that anomaly detection uses output of impact.

## Preprocessing 

Anomaly Detection Preprocessing is same with CNN Segmenter Preprocessing. If you followed the instructions and got a `training_data.hdf5` in `data/v1/decrypt/segmenter/segmented_data/${time_stamp_host}`, you can skip to the training part. 

TODO: Change how to get random walker segmentations when they can be directly pulled from pollux with dvc. 

Anomaly Detection is trained based on random walker segmentations of the dataset. These random walker segmentations are done manually and uploaded to Pollux. Please download it from pollux's `shared_data` folder, save it to `data/v1/encrypt/segmenter/random_walker_segmenter` directory and extract it.

Anomaly Detection does preprocessing of data before training the model:

First, it copies fully sampled FlowMRI data, under sampled reconstructed FlowMRI data and random walker segmentations into a work directory called `data/v1/decrypt/segmenter/segmented_data/${time_stamp_host}`.

The following script is for reorganizing the data and it takes three arguments: 
1- `time_stamp_host` of fully sampled FlowMRI data which is relative to `data/v1/decrypt/flownet/hpc_predict/v2/inference`
2- `time_stamp_host` of under sampled reconstructed FlowMRI data which is relative to `data/v1/decrypt/flownet/hpc_predict/v2/inference` 
3- `time_stamp_host` of random walker segmentations which is relative to `data/v1/decrypt/segmenter/random_walker_segmenter` 

```
./segmenter/cnn_segmenter_for_mri_4d_flow/data_flownet_reorganize_dir_structure.sh 2021-02-11_19-41-32_daint102 2021-03-19_15-46-05_daint102 2021-02-11_20-14-44_daint102

```
Secondly, these reorganized data in `segmented_data/${time_stamp_host}` directory are combined to a single hdf5 file, called `training_data.hdf5` . This output file is saved in same work directory. This stage is integrated into dvc for consistency. (TODO: decide on this file should be tracked with dvcor not. )

The first argument corresponds to output directory where container's output should be written and second argument corresponds to segmenter training work directory, relative to `segmenter/segmented_data`.

```
./flowmri_anomaly_detection/docker/dvc/dvc_run_training_preprocessing.sh data/v1 2021-05-31_11-46-39_copper 
```
We have now a dvc.yaml file in `data/v1/config/segmenter/segmented_data/${time_stamp_host}`. We can run a dvc reproducer and push the result to git and pollux.
```
cd data/v1
dvc repro -s config/segmenter/segmented_data/2021-05-31_11-46-39_copper/dvc.yaml
git add config/segmenter/segmented_data/2021-05-31_11-46-39_copper/training_data.hdf5 encrypt/segmenter/segmented_data/2021-05-31_11-46-39_copper/training_data.hdf5
git commit config/segmenter/segmented_data/2021-05-31_11-46-39_copper/training_data.hdf5 encrypt/segmenter/segmented_data/2021-05-31_11-46-39_copper/training_data.hdf5
git push
dvc push encrypt/segmenter/segmented_data/2021-05-31_11-46-39_copper/training_data.hdf5
```

## Training

After preprocessing is complete and `training_data.hdf5` file is obtained, anomaly detection can be trained. The first argument corresponds to data directory, second corresponds to segmenter training work directory where `training_data.hdf5` is stored. This script trains the anomaly detection and saves the model under `data/v1/decrypt/anomaly_detection/hpc_predict/v1/training/${time_stamp_host}`

```
./flowmri_anomaly_detection/docker/dvc/dvc_run_training.sh data/v1 2021-05-31_11-46-39_copper
```

This script will create a dvc.yaml file in `data/v1/config/anomaly_detection/hpc_predict/v1/training/output_directory` which we can run a dvc reproducer.

```
cd data/v1
dvc repro -s config/anomaly_detection/hpc_predict/v1/training/output_directory/dvc.yaml # i.e. 2021-05-31_13-34-40_copper for output_directory

```
This runs Anomaly Detection training and saves the model. Now, we can push this model to git and Pollux.

```
cd data/v1
git add config/anomaly_detection/hpc_predict/v1/training/output_directory/* encrypt/anomaly_detection/hpc_predict/v1/training/output_directory/.gitignore
git commit config/anomaly_detection/hpc_predict/v1/training/output_directory/* encrypt/anomaly_detection/hpc_predict/v1/training/output_directory/.gitignore
git push
dvc push encrypt/anomaly_detection/hpc_predict/v1/training/output_directory/
```
## Inference

After the model is obtained, anomaly detection inference can be run. This inference scripts needs four arguments:

1- dvc data directory where results will be saved 
2- Anomaly Detection Model # relative to `data/v1/decrypt/anomaly_detection/hpc_predict/v1/training/ #i.e. 2021-05-31_13-34-40_copper
3- SegmentedFlowMRI data, output of CNN Segmenter # i.e. 2021-04-20_00-33-21_copper_volN5_R10/output/kspc_R10_volN5_vn.mat_segmented.h5
4- AnomalySegmentedFlowMRI data, path to save the output of Anomaly Detection # i.e. 2021-05-28_12-17-01_copper_volN5_R10/output/kspc_R10_volN5_vn.mat_segmented_anomaly.h5

```
./flowmri_anomaly_detection/docker/dvc/dvc_run_inference.sh data/v1 2021-05-31_13-34-40_copper 2021-04-20_00-33-21_copper_volN5_R10/output/kspc_R10_volN5_vn.mat_segmented.h5 2021-05-28_12-17-01_copper_volN5_R10/output/kspc_R10_volN5_vn.mat_segmented_anomaly.h5
```

We have a dvc.yaml file in `data/v1/config/anomaly_detection/hpc_predict/v1/inference/2021-05-28_12-17-01_copper_volN5_R10` and we can run a dvc reproducer.

```
cd data/v1
dvc repro -s config/segmenter/cnn_segmenter/hpc_predict/v1/inference/output_directory/dvc.yaml # i.e. 2021-05-28_12-17-01_copper_volN5_R10 for output directory
```

This runs anomaly detection and creates AnomalySegmentedFlowMRI data, which we can push to git and Pollux. ( The small dvc-config files are pushed to git and large hdf5 ouput, AnomalySegmentedFlowMRI is pushed to Pollux.)

```
cd data/v1
git add config/anomaly_detection/hpc_predict/v1/inference/output_directory/* encrypt/anomaly_detection/hpc_predict/v1/inference/output_directory/.gitignore
git commit config/anomaly_detection/hpc_predict/v1/inference/output_directory/* encrypt/anomaly_detection/hpc_predict/v1/inference/output_directory/.gitignore
git push 
dvc push encrypt/anomaly_detection/hpc_predict/v1/inference/output_directory/output
```

If you want to run anomaly detection on all SegmentedFlowMRI data beloning to same dataset, considering they all created with same date, there is a script for it. At this moment, it accepts three arguments:
1- dvc data directory
2- Anomaly Detection Model, relative to `data/v1/decrypt/anomaly_detection/hpc_predict/v1/training/`
3- `time_stamp_host`of SegmentedFlowMRI data of volunteers and patients, i.e. #2021-04-20_00-33-21_copper

```
./flowmri_anomaly_detection/docker/dvc/dvc_run_inference_dataset.sh data/v1 2021-05-31_13-34-40_copper 2021-04-20_00-33-21_copper
```

This script will run inference for all the existing SegmentedFlowMRI data in the same dataset and sets up dvc stage file for each one of them. The output of inference will be AnomalySegmentedFlowMRI data saved in data/v1/anomaly_detection/hpc_predict/v1/inference/output_directory . Naming of output directories is done by using the run date, machine name and volunteer/patient with given R, under-sampling ratio. ( For example, 2021-05-28_12-17-01_copper_volN5_R14)

We can now run dvc reproducer for all the anomaly detection inference data and push the outputs to git and pollux.
```
cd data/v1
dvc repro -s config/anomaly_detection/v1/inference/2021-05-28_12-17-01_copper_*/dvc.yaml
git add config/anomaly_detection/hpc_predict/v1/inference/2021-05-28_12-17-01_copper_* encrypt/anomaly_detection/hpc_predict/v1/inference/2021-05-28_12-17-01_copper_*/.gitignore
git commit config/anomaly_detection/hpc_predict/v1/inference/2021-05-28_12-17-01_copper_* encrypt/anomaly_detection/hpc_predict/v1/inference/2021-05-28_12-17-01_copper_*/.gitignore
git push
dvc push encrypt/anomaly_detection/hpc_predict/v1/inference/2021-05-28_12-17-01_copper_*/output
```
 
