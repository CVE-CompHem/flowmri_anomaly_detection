# ==================================================================
# import modules
# ==================================================================
import utils
import os, sys
import numpy as np
import tensorflow as tf
from args import args

# ==================================================================
# import experiment settings
# ==================================================================
from experiments.unet import model_config as exp_config

# ==================================================================
# import models
# ==================================================================
from networks.variational_autoencoder import VariationalAutoencoder
from models.vae import VAEModel

# ==================================================================
# import general modules written by cscs for the hpc-predict project
# ==================================================================
current_dir_path = os.getcwd()
mr_io_dir_path = current_dir_path[:-25] + 'hpc-predict-io/python/'
sys.path.append(mr_io_dir_path)
from mr_io import SegmentedFlowMRI

# ==================================================================
# import and setup logging
# ==================================================================
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# set loggin directory
# ==================================================================
log_dir = args.training_output + exp_config.model_name
logging.info('Logging directory: %s' %log_dir)

# ============================================================================
# input: image with shape  ---> [nx, ny, nz, nt, 4]
# output: segmentation probability of the aorta with shape  ---> [nx, ny, nz, nt]
# ============================================================================
def get_vae_recon(test_image,
                  model):
        
    # create an empty array to store the VAE recon
    recon = np.zeros((test_image.shape))
    
    # predict the segmentation 8 zz slices at a time (this is the batch size parameter in the config file)
    for zz in range(test_image.shape[2]):
        feed_dict = {model.image_matrix: np.expand_dims(test_image[:,:,zz,:,:], 0)}
        out_mu = model.sess.run(model.decoder_output, feed_dict)
        recon[:,:,zz,:,:] = np.squeeze(out_mu)
        
    return recon

# ==================================================================
# main function for inference
# ==================================================================
def run_inference():

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
   # logging.info('================ EXPERIMENT NAME: %s ================' % exp_config.model_name)
    logging.info('================ Detecting anomaly for this image: %s ================' % args.inference_input)

    # ============================
    # load saved checkpoint file, if available
    # ============================
    logging.info('================ Looking for saved segmentation model... ')
   # model_iter = 2000
    #logging.info('args.training_output: ' + args.training_output + exp_config.model_name + '/models/model.ckpt-' + str(model_iter))
    if os.path.exists(args.training_output + '/model.ckpt-2000'+ '.index'):
    #    best_model_checkpoint_path = args.training_output + exp_config.model_name + '/models/model.ckpt-' + str(model_iter)
        best_model_checkpoint_path = args.training_output + '/model.ckpt-2000'
        logging.info('Found saved model at %s. This will be used for predicted the segmentation.' % best_model_checkpoint_path)
    else:
        logging.warning('Did not find a saved model. First need to run training successfully...')
        raise RuntimeError('No checkpoint available to restore from!')

    # ============================   
    # Loading data (a SegmentedFlowMRI object written by IMPACT / segmentation CNN and in the filename given by inference_input)
    # ============================   
    logging.info('================ Loading input SegmentedFlowMRI from: ' + args.inference_input)    
    segmentedflow_mri = SegmentedFlowMRI.read_hdf5(args.inference_input)
    segmentedflow_mri_image = np.concatenate([np.expand_dims(segmentedflow_mri.intensity, -1), segmentedflow_mri.velocity_mean], axis=-1)  
    segmentedflow_mri_label = np.expand_dims(segmentedflow_mri.segmentation_prob, -1)
    logging.info('shape of the test image before cropping / padding: ' + str(segmentedflow_mri_image.shape))
    # normalize
    segmentedflow_mri_image = utils.normalize_image(segmentedflow_mri_image)
    # mask the image using the segmentations
    label_binary = np.round(segmentedflow_mri_label)
    label_binary_tiled = np.tile(label_binary, (1, 1, 1, 1, 4))
    segmentedflow_mri_image = label_binary_tiled * segmentedflow_mri_image
    # crop / pad to common size
    orig_volume_size = segmentedflow_mri_image.shape[0:4]
    common_volume_size = [112, 112, 24, 24]
    segmentedflow_mri_image_cropped = utils.crop_or_pad_4dvol(segmentedflow_mri_image, common_volume_size)
    logging.info('shape of the test image after cropping / padding: ' + str(segmentedflow_mri_image_cropped.shape))
        
    # ============================
    # build the TF graph
    # ============================
    with tf.Graph().as_default():
    
        vae_network = VariationalAutoencoder

        model = VAEModel(vae_network,
                         exp_config,
                         exp_config.model_name,
                         log_dir)
    
        # ============================
        # Load the trained vae model 
        # ============================
        model.load_from_path(best_model_checkpoint_path)
        
        # ============================
        # predict the segmentation probability for the image
        # ============================        
        reconstructed_image = get_vae_recon(segmentedflow_mri_image_cropped, model)
        logging.info('shape of reconstruction image: ' + str(reconstructed_image.shape))

        # ============================
        # crop / pad back to the original dimensions
        # ============================
        reconstructed_image = utils.crop_or_pad_4dvol(reconstructed_image, orig_volume_size)
        logging.info('shape of reconstruction error after cropping back to original size: ' + str(reconstructed_image.shape))
        
        # ============================
        # create an instance of the SegmentedFlowMRI class, with the image information from flow_mri as well as the predicted anomaly probabilities
        # ============================
        anomalous_flow_mri = SegmentedFlowMRI(segmentedflow_mri.geometry,
                                              segmentedflow_mri.time,
                                              segmentedflow_mri.time_heart_cycle_period,
                                              reconstructed_image[:,:,:,:,0],
                                              reconstructed_image[:,:,:,:,1:4],
                                              segmentedflow_mri.velocity_cov,
                                              segmentedflow_mri.segmentation_prob)

        # ============================
        # write SegmentedFlowMRI to file
        # ============================
        logging.info('================ Writing SegmentedFlowMRI to: ' + args.inference_output)
        anomalous_flow_mri.write_hdf5(args.inference_output)
        logging.info('============================================================')

# ==================================================================
# ==================================================================
def main():    
    run_inference()

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
