import numpy as np

from IPython.display import SVG
from keras.layers import Concatenate, Input, Dense, Reshape, Flatten, Dropout, multiply, \
    BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot, plot_model
from matplotlib import pyplot

 
class GAN():
    
    def __init__(self, in_shape=(28,28,1), n_classes=10, latent_dim=100):
        # size of the latent space
        self.in_shape = in_shape
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.optimizer = Adam(lr=0.0002, beta_1=0.5) 
        
        # create the discriminator
        self.d_model = self.build_discriminator()
        
        # create the generator
        self.g_model = self.build_generator()
        
        # create the gan
        self.gan_model = self.build_gan()
       
    def build_discriminator(self):
        """define the standalone discriminator model"""
        pass
    
    def build_generator(self):
        """define the standalone generator model"""
        pass

    def build_gan(self):
        """define the combined generator and discriminator model, for updating the generator"""
        pass 
    
    @staticmethod
    def plot_model(model, filename=None):
        """Plot the specified model, either to file, or return as an SVG if no filename is specified"""
        if filename is None:
            return(SVG(model_to_dot(model, show_shapes=True, show_layer_names=True).create(prog='dot', format='svg')))
        else:
            plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
            
    def plot_generator(self, filename=None):
        """Plot the generator model, either to file, or return as an SVG if no filename is specified"""
        return(self.plot_model(self.g_model))
    
    def plot_discriminator(self, filename=None):
        """Plot the discriminator model, either to file, or return as an SVG if no filename is specified"""
        return(self.plot_model(self.d_model))
                
    def plot_gan(self, filename=None):
        """Plot the gan model, either to file, or return as an SVG if no filename is specified"""
        return(self.plot_model(self.g_model))
    
    def generate_real_samples(self, X, n_samples):
        """select real samples"""
        # choose random instances
        ix = np.random.randint(0, X.shape[0], n_samples)
        # select images and labels
        selected_X = X[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, 1))
        return selected_X, y
    
    def generate_latent_points(self, n_samples):
        """generate points in latent space as input for the generator"""
        # generate points in the latent space
        latent_points = np.random.randn(self.latent_dim * n_samples)
        # reshape into a batch of inputs for the network
        latent_points = latent_points.reshape(n_samples, self.latent_dim)
        return latent_points
    
    def generate_generator_prediction_samples(self, n_samples):
        """use the generator to generate n fake examples"""
        # generate points in latent space
        latent_points = self.generate_latent_points(n_samples)
        # predict outputs
        X = self.generator_prediction(latent_points)
        # create class labels
        y = np.zeros((n_samples, 1))
        return X, y
    
    def generator_prediction(self, latent_points):
        """use the generator to predict outputs for the given latent space"""
        return self.g_model.predict(latent_points)

    def train(self, X, labels, n_epochs=100, n_batch=128, reporting_period=10):
        """train the generator and discriminator"""
        pass

    def summarize_performance(self, epoch, filename_base):
        """generate samples and save as a plot and save the model"""
        self.plot_random_generated_images()
        
        # save the plot
        filename1 = '{}_generated_plot_{:04d}.png'.format(filename_base, epoch)
        pyplot.savefig(filename1)
        pyplot.close()
        
        # save the generator model
        filename2 = '{}_model_{:04d}.h5'.format(filename_base, epoch)
        self.g_model.save(filename2)
        print('>Saved: %s and %s' % (filename1, filename2))
        
    @staticmethod
    def grid_plot(examples, dimensions=(10, 10), figsize=(10, 10), cmap='gray_r'):
        """create a grid plot of multiple images"""
        pyplot.figure(figsize=figsize)
        for i in range(dimensions[0] * dimensions[1]):
            pyplot.subplot(dimensions[0], dimensions[1], 1 + i)
            pyplot.axis('off')
            pyplot.imshow(np.squeeze(examples[i]), interpolation='nearest', cmap=cmap)
        pyplot.tight_layout()
        pyplot.show()

    def plot_random_generated_images(self):
        """create a plot of randomly generated images (reversed grayscale)"""
        dimensions=(10, 10)
        figsize=(10, 10)
        n_samples=100
        
        X, _ = self.generate_generator_prediction_samples(n_samples)
            
        self.grid_plot(X, dimensions=dimensions, figsize=figsize)

        #filename = '%s_generated_plot_e%03d.png' % (identifier, epoch+1)
        #pyplot.savefig(filename)
        #pyplot.close()
    
    def plot_training_metrics(self, figsize = (10,10)):
        fig = pyplot.figure(figsize=figsize)
        
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.plot(self.d_loss_real, label='Discriminator Loss (real)')
        pyplot.plot(self.d_loss_fake, label='Discriminator Loss (fake)')
        pyplot.plot(self.g_loss, label='Generator Loss')
        pyplot.legend(loc='upper right')
        pyplot.title('Losses')

        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.plot(self.d_acc_real, label='Discriminator Accuracy (real)')
        pyplot.plot(self.d_acc_fake, label='Discriminator Accuracy (fake)')
        pyplot.ylabel('')
        pyplot.xlabel('batch')
        pyplot.legend(loc='upper right')
        pyplot.title('Accuracies')

        pyplot.tight_layout(rect=[0, 0.00, 1, 0.95])
        fig.suptitle('GAN Per Batch Training Metrics', fontsize=24)  
        pyplot.show()
        
        # save plot to file
        # pyplot.savefig('results_baseline/plot_line_plot_loss.png')
        pyplot.close()

    def plot_discriminator_accuracies(self):
        pyplot.plot(self.d_acc_real_epochs)
        pyplot.plot(self.d_acc_fake_epochs)
        pyplot.title('GAN Per Epoch Discriminator Accuracy')
        pyplot.ylabel('')
        pyplot.xlabel('epoch')
        pyplot.legend(['Discriminator (real)', 'Discriminator (fake)'],loc='upper right')
        pyplot.show()
        
    def save_model(self, filename_base):
        """save the model"""
        self.g_model.save('{}_g_model.h5'.format(filename_base))
        self.d_model.save('{}_d_model.h5'.format(filename_base))
        self.gan_model.save('{}_gan_model.h5'.format(filename_base))
       
    def load_model(self, filename_base):
        """load the model"""
        self.g_model = load_model('{}_g_model.h5'.format(filename_base))
        self.d_model = load_model('{}_d_model.h5'.format(filename_base))
        self.gan_model = load_model('{}_gan_model.h5'.format(filename_base))        