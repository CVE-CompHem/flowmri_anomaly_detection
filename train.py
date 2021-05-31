# ==================================================================
# import modules
# ==================================================================
import shutil
import h5py
import numpy as np
import tensorflow as tf

# ==================================================================
# import models
# ==================================================================
from networks.variational_autoencoder import VariationalAutoencoder
from models.vae import VAEModel

# arguments
from args import args

# ==================================================================
# The config parameters are imported below
# This is done is a somewhat (and perhaps, unnecessarily complicated) manner!
# First, we look into the 'unet.py' file that is present inside the experiments directory
# This, in turn, reads the model parameters from args.model file, which, in turn, is set in the args.py file(!)
# Currently, the args.model is set to 'experiments/unet.json' file. 
# So, ultimately, the parameters that are read below are from the experiments/unet_neerav.json file.
# ==================================================================
from experiments.unet import model_config as exp_config

# ==================================================================
# import and setup logging
# ==================================================================
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ===================================
# Set the logging directory
# ===================================
log_dir = args.training_output + "/" + exp_config.model_name
logging.info('Logging directory: %s' %log_dir)

# ==================================================================
# main function for training
# ==================================================================
def run_training(continue_run):

    # ============================
    # log experiment details
    # ============================
    logging.info('============================================================')
    logging.info('EXPERIMENT NAME: %s' % exp_config.model_name)

    # ============================
    # Load training data
    # ============================   
    logging.info('============================================================')
    logging.info('Loading training data')    
    training_data_hdf5 = h5py.File(args.training_input, "r")
    images_tr = training_data_hdf5['images'] # e.g.[21, 112, 112, 20, 25, 4] : [num_subjects*num_r_values, nx, ny, nz, nt, 4]
    labels_tr = training_data_hdf5['labels'] # e.g.[21, 112, 112, 20, 25, 1] : [num_subjects*num_r_values, nx, ny, nz, nt, 1] # contains the prob. of the aorta (FG)
    logging.info('Shape of training images: %s' %str(images_tr.shape)) 
    logging.info('Shape of training labels: %s' %str(labels_tr.shape))
    logging.info('============================================================')
            
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ============================
        # set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(exp_config.run_number)
        np.random.seed(exp_config.run_number)

        # ====================================================================================
        # Initialize the network architecture, training parameters, model_name, and logging directory
        # ====================================================================================
        vae_network = VariationalAutoencoder
        model = VAEModel(vae_network,
                         exp_config,
                         exp_config.model_name,
                         log_dir)
        model.initialize()
        model.summarize()
        
        logging.info('============================================================')
        logging.info('Starting training iterations....')
        for iter_ in range(exp_config.max_iterations):
            
            # lambda for the KL divergence loss
            weight = 1.
            
            x, y = get_batch(images_tr,
                             labels_tr,
                             exp_config)

            # mask the images to remove the background
            y_binary = np.round(y[:,:,:,:,1])
            y_binary = np.expand_dims(y_binary, axis = -1)
            y_binary_tiled = np.tile(y_binary, (1, 1, 1, 1, 4))
            x = y_binary_tiled * x
            
            # run the training step
            model.train(x, weight)

            if iter_ % exp_config.summary_writing_frequency == 0:

                logging.info("Iteration:" + str(iter_))
                
                x = x.astype("float32")

                # Write summary for tensorboard
                summary_str = model.sess.run(model.summary_op, {model.image_matrix: x, model.weight: weight})
                model.writer.add_summary(summary_str, iter_)
                model.writer.flush()
                
                gen_loss, res_loss, lat_loss = model.sess.run([model.autoencoder_loss,
                                                               model.autoencoder_res_loss,
                                                               model.latent_loss], {model.image_matrix: x})            

                logging.info(("Iteration %d: train_gen_loss %f train_lat_loss %f train_res_loss %f total train_loss %f") % (
                        iter_, gen_loss.mean(), lat_loss.mean(), res_loss.mean(), 100.*gen_loss.mean()+lat_loss.mean()))

            if iter_ % exp_config.save_frequency == 0:
                model.save(iter_)

        # close the tf session
        model.sess.close() 

        # close the hdf5 file containing the training data
        training_data_hdf5.close()

# ==================================================================
# ==================================================================
def get_batch(images,
              labels,
              batch_size):
    '''
    Function to get a batch from the dataset
    :param images: numpy array
    :param labels: numpy array
    :param batch_size: batch size
    :return: batch
    '''

    x = np.zeros((exp_config.batch_size,
                  exp_config.image_size[0],
                  exp_config.image_size[1],
                  exp_config.image_size[2],
                  exp_config.nchannels), dtype = np.float32)
    
    y = np.zeros((exp_config.batch_size,
                  exp_config.image_size[0],
                  exp_config.image_size[1],
                  exp_config.image_size[2],
                  2), dtype = np.float32)
    
    for b in range(exp_config.batch_size):  
    
        # ===========================
        # generate indices to randomly select different x-y-t volumes in the batch
        # ===========================
        random_image_index = np.random.randint(images.shape[0])
        random_z_index = np.random.randint(images.shape[3])
        
        x[b, :, :, :, :] = images[random_image_index, :, :, random_z_index, :, :]
        y[b, :, :, :, 0] = 1 - labels[random_image_index, :, :, random_z_index, :, 0] # prob. of background
        y[b, :, :, :, 1] = labels[random_image_index, :, :, random_z_index, :, 0] # prob. of foreground
    
    return x, y

# ==================================================================
# ==================================================================
def train():
    
    # ===========================
    # Create dir if it does not exist
    # ===========================
    continue_run = exp_config.continue_run
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
        tf.gfile.MakeDirs(log_dir + '/models')
        continue_run = False

    # ===========================
    # Copy experiment config file
    # ===========================
    shutil.copy(args.model, log_dir) # exp_config.__file__

    # ===========================
    # run training
    # ===========================
    run_training(continue_run)

# ==================================================================
# ==================================================================
def main():
    if args.debug_server is not None:
        try:
            import pydevd_pycharm
            debug_server_hostname, debug_server_port = args.debug_server.split(':')
            pydevd_pycharm.settrace(debug_server_hostname,
                                    port=int(debug_server_port), 
                                    stdoutToServer=True,
                                    stderrToServer=True)
        except:
            logging.error("Import error for pydevd_pycharm ignored (should not be running debug version).")

    # ===========================
    # Run the training
    # ===========================
    train()

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
