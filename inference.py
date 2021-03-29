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
def get_anomaly_probability(test_image,
                            images_pl,
                            training_pl,
                            seg_prob_op,
                            sess):
        
    # create an empty array to store the predicted segmentation probabilities
    predicted_seg_prob_aorta = np.zeros((test_image.shape[:-1]))
    
    # predict the segmentation one zz slice at a time
    for zz in range(test_image.shape[2]):
        predicted_seg_prob = sess.run(seg_prob_op, feed_dict = {images_pl: np.expand_dims(test_image[:,:,zz,:,:], axis=0), training_pl: False})
        predicted_seg_prob = np.squeeze(predicted_seg_prob) # squeeze out the added batch dimension
        predicted_seg_prob_aorta[:,:,zz,:] = predicted_seg_prob[:, :, :, 1] # the prob. of the FG is in the second channel
        
    return predicted_seg_prob_aorta

# ==================================================================
# main function for inference
# ==================================================================
def run_inference():

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
    logging.info('================ EXPERIMENT NAME: %s ================' % exp_config.model_name)
    logging.info('================ Detecting anomaly for this image: %s ================' % args.inference_input)

    # ============================
    # load saved checkpoint file, if available
    # ============================
    logging.info('================ Looking for saved segmentation model... ')
    model_iter = 2000
    logging.info('args.training_output: ' + args.training_output + exp_config.model_name + '/models/model.ckpt-' + str(model_iter))
    if os.path.exists(args.training_output + exp_config.model_name + '/models/model.ckpt-' + str(model_iter) + '.index'):
        best_model_checkpoint_path = args.training_output + exp_config.model_name + '/models/model.ckpt-' + str(model_iter)
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
    segmentedflow_mri_image = utils.crop_or_pad_4dvol(segmentedflow_mri_image, common_volume_size)
    logging.info('shape of the test image after cropping / padding: ' + str(segmentedflow_mri_image.shape))
        
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
        # create an empty array to store the predicted anomaly probabilities
        recon_error = np.zeros((segmentedflow_mri_image.shape))
        
        # predict the segmentation 8 zz slices at a time (this is the batch size parameter in the config file)
        for zz in range(segmentedflow_mri_image.shape[2]):
            feed_dict = {model.image_matrix: np.expand_dims(segmentedflow_mri_image[:,:,zz,:,:], 0)}
            out_mu = model.sess.run(model.decoder_output, feed_dict)
            recon_error[:,:,zz,:,:] = segmentedflow_mri_image[:,:,zz,:,:] - np.squeeze(out_mu)

        logging.info('shape of reconstruction error: ' + str(recon_error.shape))

        # ============================
        # crop / pad back to the original dimensions
        # ============================
        recon_error = utils.crop_or_pad_4dvol(recon_error, orig_volume_size)
        logging.info('shape of reconstruction error after cropping back to original size: ' + str(recon_error.shape))
        
        # ============================
        # compute rmse along the channel direction
        # ============================
        recon_error_rmse_along_channel = np.sqrt(np.sum(np.square(recon_error), -1))
        logging.info('shape of reconstruction error rmse along channel direction: ' + str(recon_error_rmse_along_channel.shape))

        # ============================
        # create an instance of the SegmentedFlowMRI class, with the image information from flow_mri as well as the predicted anomaly probabilities
        # ============================
        anomalous_flow_mri = SegmentedFlowMRI(segmentedflow_mri.geometry,
                                              segmentedflow_mri.time,
                                              segmentedflow_mri.time_heart_cycle_period,
                                              segmentedflow_mri.intensity,
                                              segmentedflow_mri.velocity_mean,
                                              segmentedflow_mri.velocity_cov,
                                              recon_error_rmse_along_channel)

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