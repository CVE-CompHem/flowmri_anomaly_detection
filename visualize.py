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
from mr_io import FlowMRI, SegmentedFlowMRI

def norm(x,y,z):
    normed_array = np.sqrt(np.square(x)+np.square(y)+np.square(z))
    return normed_array 

basepath = '/usr/bmicnas01/data-biwi-01/nkarani/projects/hpc_predict/data/eth_ibt/flownet/pollux/all_data/'

# plot
p_or_v = 'v' # patient or volunteer
R_values = [8] # undersampling ratio
for subnum in [4]: # subject number (change this for plotting patient results)

    for r in range(len(R_values)):

        R = R_values[r]
        
        imagepath = basepath + str(p_or_v) + str(subnum) + '_R' + str(R) + '.h5'
        segmentationpath = basepath + str(p_or_v) + str(subnum) + '_R' + str(R) + '_seg_cnn.h5'
        anomalypath = basepath + str(p_or_v) + str(subnum) + '_R' + str(R) + '_anomaly.h5'
            
        # read the flownet image
        flow_mri = FlowMRI.read_hdf5(imagepath)
        image = np.concatenate([np.expand_dims(flow_mri.intensity, -1), flow_mri.velocity_mean], axis=-1)
        image = utils.normalize_image(image)
        
        # read the segmentation
        segmented_flow_mri = SegmentedFlowMRI.read_hdf5(segmentationpath)
        segmentation = np.expand_dims(segmented_flow_mri.segmentation_prob, -1)
        
        # mask the image using the segmentations
        segmentation_binary = np.round(segmentation)
        segmentation_binary_tiled = np.tile(segmentation_binary, (1, 1, 1, 1, 4))
        image_masked = segmentation_binary_tiled * image
        
        # read the reconstructed image
        segmented_flow_mri = SegmentedFlowMRI.read_hdf5(anomalypath)
        recon_image = np.concatenate([np.expand_dims(segmented_flow_mri.intensity, -1), segmented_flow_mri.velocity_mean], axis=-1)
        recon_image_masked = segmentation_binary_tiled * recon_image
        
        # compute the anomaly as the reconstruction error inside the masked aorta
        reconstruction_error = recon_image_masked - image_masked
        reconstruction_error_rmse_along_channel = np.sqrt(np.sum(np.square(reconstruction_error), -1))
        anomaly = reconstruction_error_rmse_along_channel
        
        # threshold the anomaly (how to define a threshold?)
        threshold = 1.0
        anomaly[anomaly >= threshold] = 1.0
        anomaly[anomaly < threshold] = 0.0
        
        nr = 2
        nc = 3
        
        for idx in range(19):
            
            plt.figure(figsize=[5*nc, 5*nr])
            
            zidx = idx
            # across z axis
            plt.subplot(nr, nc, 1);  plt.imshow(norm(image[:,:,zidx,3,1], image[:,:,zidx,3,2], image[:,:,zidx,3,3]), cmap='gray'); plt.axis('off'); plt.title('image_z'+str(zidx)+'_t3'); plt.clim([0.0, 1.6]); plt.colorbar()
            plt.subplot(nr, nc, 2);  plt.imshow(norm(recon_image[:,:,zidx,3,1], recon_image[:,:,zidx,3,2], recon_image[:,:,zidx,3,3]), cmap='gray'); plt.axis('off'); plt.title('vae_recon_z'+str(zidx)+'_t3'); plt.clim([0.0, 1.6]); plt.colorbar()
            plt.subplot(nr, nc, 3);  plt.imshow(anomaly[:,:,zidx,3], cmap='gray'); plt.axis('off'); plt.title('anomaly z'+str(zidx)+'_t3'); plt.colorbar()

            # across t axis
            plt.subplot(nr, nc, 4);  plt.imshow(norm(image[:,:,8,idx,1], image[:,:,8,idx,2], image[:,:,8,idx,3]), cmap='gray'); plt.axis('off'); plt.title('image_z8_t'+str(idx)); plt.clim([0.0, 1.6]); plt.colorbar()
            plt.subplot(nr, nc, 5);  plt.imshow(norm(recon_image[:,:,8,idx,1], recon_image[:,:,8,idx,2], recon_image[:,:,8,idx,3]), cmap='gray'); plt.axis('off'); plt.title('vae_recon_z8_t'+str(idx)); plt.clim([0.0, 1.6]); plt.colorbar()
            plt.subplot(nr, nc, 6);  plt.imshow(anomaly[:,:,8,idx], cmap='gray'); plt.axis('off'); plt.title('anomaly z8_t'+str(idx)); plt.colorbar()
                    
            plt.savefig(basepath + 'tmp/pngs' + str(idx) + '.png')
            plt.close()
            
        _gif = []
        for idx in range(19):
            _gif.append(imageio.imread(basepath + 'tmp/pngs' + str(idx) + '.png'))
        imageio.mimsave(basepath + str(p_or_v) + str(subnum) + '_R' + str(R) + '_anomaly_threshold' + str(threshold) + '.gif', _gif, format='GIF', duration=0.75)
