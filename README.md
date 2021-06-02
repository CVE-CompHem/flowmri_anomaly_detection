## Anomaly Detection

# Anomaly Detection Training

Anomaly detection does some preprocessing on data before training the model. This preprocessing is same with CNN segmenter's. If you followed the instructions for CNN segmenter, you can directly use its directory, so skip to fourth step. Otherwise, please follow all the steps. 

Steps for running the training on Flownet dataset:
1. Pull Flownet data (fully-sampled as well as at different undersampling ratios) from Pollux.
2. Pull Random walker segmentations from Pollux (these serve as ground truths for training the anomaly detection model).
3. Run the `segmenter/cnn_segmenter/data_flownet_reorganize_dir_structure.sh` script. This will simplify the directory structure and file names.
   
This script takes three arguments, time_stamp_host of fully sampled Flownet FlowMRI data, time_stamp_host of under-sampled Flownet FlowMRI data and time_stamp_host of random walker segmentations of Flownet dataset. 

```
./segmenter/cnn_segmenter/data_flownet_reorganize_dir_structure.sh 2021-02-11_19-41-32_daint102 2021-03-19_15-46-05_daint102 2021-02-11_20-14-44_daint102
```

This will create a directory under `data/v1/decrypt/segmenter/segmented_data/${time_stamp_host}` and copy the necessary files there. 

4. Run the `segmenter/cnn_segmenter_for_mri_4d_flow/data_flownet_prepare_training_data.py` script. This will combine the training data and labels into one hdf5 file called `training_data.hdf5` . 
   This script takes one argument which indicates the the working directory, i.e. `data/v1/decrypt/segmenter/segmented_data/${time_stamp_host}`.

``` 
python segmenter/cnn_segmenter/data_flownet_prepare_training_data.py data/v1/decrypt/segmenter/segmenter_data/2021-05-31_11-46-39_copper 
``` 
 
5. Run the training command as given below.

```
python flowmri_anomaly_detection/train.py --training_input <path to hdf5 file containing the training data> # i.e data/v1/decrypt/segmenter/segmented_data/2021-05-31_11-46-39_copper/training_data.hdf5 --training_output <path where the training CNN model should be saved> # data/v1/decrypt/segmenter/cnn_segmenter/hpc_predict/v1/training/2021-05-31_20-00-00_copper_flownet
```

# Anomaly Detection Inference

To run inference, run the following command:

```
python flowmri_anomaly_detection/inference.py --training_output <path where the training model is saved> # i.e. data/v1/decrypt/segmenter/cnn_segmenter/hpc_predict/v1/training/2021-05-31_20-00-00_copper_flownet --inference_input <path of the SegmentedFlowMRI data> # i.e. data/v1/decrypt/segmenter/cnn_segmenter/hpc_predict/v1/inference/2021-04-20_00-33-21_copper_volN7_R18/output/kspc_R18_volN7_vn.mat_segmented.h5 --inference_output <path where the AnomalySegmentedFlowMRI data should be saved> # i.e. data/v1/decrypt/anomaly_detection/hpc_predict/v1/inference/2021-05-28_12-17-01_copper_volN5_R14/output/kspc_R14_volN5_vn.mat_segmented_anomaly.h5
```

