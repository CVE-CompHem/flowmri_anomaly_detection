#!/bin/bash

time_stamp_host_input=$3

time_stamp_host_output=$(date +'%Y-%m-%d_%H-%M-%S')_$(hostname)
for N in 4 5 6 7	
do
    srcdir=${time_stamp_host_input}
    srcdir+="_volN"
    srcdir+=$N
    srcdir+="_R1"
    srcdir+="/output/recon_volN"
    srcdir+=$N
    srcdir+="_vn.mat_segmented.h5"

    dstdir=${time_stamp_host_output}
    dstdir+="_volN"
    dstdir+=$N
    dstdir+="_R1"
    
    sh ./flowmri_anomaly_detection/docker/dvc/dvc_run_inference.sh $1 $2 $srcdir $dstdir

done

for N in 4 5 6 7
do
    for R in 8 10 12 14 16 18 20 22
    do
	srcdir=${time_stamp_host_input}
	srcdir+="_volN"
	srcdir+=$N
	srcdir+="_R"
	srcdir+=$R
	srcdir+="/output/kspc_R"
	srcdir+=$R
	srcdir+="_volN"
	srcdir+=$N
	srcdir+="_vn.mat_segmented.h5"

	dstdir=${time_stamp_host_output}
	dstdir+="_volN"
	dstdir+=$N
	dstdir+="_R"
	dstdir+=$R

	sh ./flowmri_anomaly_detection/docker/dvc/dvc_run_inference.sh $1 $2 $srcdir $dstdir
    done
done

for R in 8 10 12 14
do
    srcdir=${time_stamp_host_input}
    srcdir+="_patient1_R"
    srcdir+=$R
    srcdir+="/output/kspc_R"
    srcdir+=$R
    srcdir+="_vn.mat_segmented.h5"

    dstdir=${time_stamp_host_output}
    dstdir+="_patient1_R"
    dstdir+=$R

    sh ./flowmri_anomaly_detection/docker/dvc/dvc_run_inference.sh $1 $2 $srcdir $dstdir
done
