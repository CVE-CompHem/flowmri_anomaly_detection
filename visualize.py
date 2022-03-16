# import module and set paths
# ============================   
import os, sys
import numpy as np
import utils
import imageio
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})

# ==================================================================
# import general modules written by cscs for the hpc-predict project
# ==================================================================
current_dir_path = os.getcwd()
mr_io_dir_path = current_dir_path[:-25] + 'hpc-predict-io/python/'
sys.path.append(mr_io_dir_path)
from hpc_predict_io.mr_io import FlowMRI, SegmentedFlowMRI, AnomalySegmentedFlowMRI

def norm(x,y,z):
    normed_array = np.sqrt(np.square(x)+np.square(y)+np.square(z))
    return normed_array 

basepath = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/eth_ibt/flownet/pollux/all_data/'

# ===================================
# plot
# ===================================
p_or_v = 'v' # patient or volunteer
R_values = [8] # undersampling ratio
for subnum in [7]: # subject number (change this for plotting patient results)

    for r in range(len(R_values)):

        R = R_values[r]

        # ===================================
        # read the HDF5 file 
        # ===================================
        anomalypath = basepath + str(p_or_v) + str(subnum) + '_R' + str(R) + '_anomaly.h5'
        anomaly_segmented_flow_mri = AnomalySegmentedFlowMRI.read_hdf5(anomalypath)

        # ===================================
        # Extract image (intensity and velocity data)
        # ===================================
        image = np.concatenate([np.expand_dims(anomaly_segmented_flow_mri.intensity, -1), anomaly_segmented_flow_mri.velocity_mean], axis=-1)
        image = utils.normalize_image(image)
        
        # ===================================
        # Extract the segmentation probability
        # ===================================
        segmentation = anomaly_segmented_flow_mri.segmentation_prob
        
        # ===================================
        # Extract the anomaly probability
        # ===================================
        anomaly = anomaly_segmented_flow_mri.anomaly_prob
        
        nr = 2
        nc = 3
        
        for idx in range(19):
            
            plt.figure(figsize=[5*nc, 5*nr])
            
            # ===================================
            # See variation along z at a fixed t value
            # ===================================
            zidx = idx
            tidx = 3
            title_suffix = '_z' + str(zidx) + '_t' + str(tidx)
            overlay_factor = 0.9 # determines the strength of the overlay of the segmentation / anomaly probabilities
            image_2d = norm(image[:,:,zidx,tidx,1], image[:,:,zidx,tidx,2], image[:,:,zidx,tidx,3]) # velocity magnitude
            seg_prob_2d = segmentation[:,:,zidx,tidx] # segmentation probability
            ano_prob_2d = anomaly[:,:,zidx,tidx] # anomaly probability
            
            # see velocity magnitude
            plt.subplot(nr, nc, 1)
            plt.imshow(image_2d, cmap='gray')
            plt.axis('off')
            plt.title('image' + title_suffix)
            # plt.clim([0.0, 1.6])
            plt.colorbar()
            
            # see segmentation overlay on the velocity magnitude
            plt.subplot(nr, nc, 2)
            plt.imshow((1-overlay_factor)*image_2d + overlay_factor*seg_prob_2d, cmap='gray')
            plt.axis('off')
            plt.title('seg_prob' + title_suffix)
            # plt.clim([0.0, 1.6])
            plt.colorbar()

            # see anomaly overlay on the velocity magnitude
            plt.subplot(nr, nc, 3)
            plt.imshow((1-overlay_factor)*image_2d + overlay_factor*ano_prob_2d, cmap='gray')
            plt.axis('off')
            plt.title('ano_prob' + title_suffix)
            # plt.clim([0.0, 1.6])
            plt.colorbar()

            # ===================================
            # See variation along t at a fixed z value
            # ===================================
            zidx = 8
            tidx = idx
            title_suffix = '_z' + str(zidx) + '_t' + str(tidx)
            image_2d = norm(image[:,:,zidx,tidx,1], image[:,:,zidx,tidx,2], image[:,:,zidx,tidx,3]) # velocity magnitude
            seg_prob_2d = segmentation[:,:,zidx,tidx] # segmentation probability
            ano_prob_2d = anomaly[:,:,zidx,tidx] # anomaly probability
            
            # see velocity magnitude
            plt.subplot(nr, nc, 4)
            plt.imshow(image_2d, cmap='gray')
            plt.axis('off')
            plt.title('image' + title_suffix)
            # plt.clim([0.0, 1.6])
            plt.colorbar()
            
            # see segmentation overlay on the velocity magnitude
            plt.subplot(nr, nc, 5)
            plt.imshow((1-overlay_factor)*image_2d + overlay_factor*seg_prob_2d, cmap='gray')
            plt.axis('off')
            plt.title('seg_prob' + title_suffix)
            # plt.clim([0.0, 1.6])
            plt.colorbar()

            # see anomaly overlay on the velocity magnitude
            plt.subplot(nr, nc, 6)
            plt.imshow((1-overlay_factor)*image_2d + overlay_factor*ano_prob_2d, cmap='gray')
            plt.axis('off')
            plt.title('ano_prob' + title_suffix)
            # plt.clim([0.0, 1.6])
            plt.colorbar()
                    
            plt.savefig(basepath + 'tmp/pngs' + str(idx) + '.png')
            plt.close()
            
        _gif = []
        for idx in range(19):
            _gif.append(imageio.imread(basepath + 'tmp/pngs' + str(idx) + '.png'))
        imageio.mimsave(basepath + str(p_or_v) + str(subnum) + '_R' + str(R) + '_anomaly.gif', _gif, format='GIF', duration=0.75)
