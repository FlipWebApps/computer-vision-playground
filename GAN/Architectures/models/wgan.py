import numpy as np
from keras import backend
from keras.constraints import Constraint
from keras.initializers import RandomNormal
from keras.layers import Conv2DTranspose, BatchNormalization, Dense, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.optimizers import RMSprop

from models.gan import GAN

class WGAN(GAN):
    # clip model weights to a given hypercube
    class ClipConstraint(Constraint):
        # set clip value when initialized
        def __init__(self, clip_value):
            self.clip_value = clip_value

        # clip model weights to hypercube
        def __call__(self, weights):
            return backend.clip(weights, -self.clip_value, self.clip_value)

        # get the config
        def get_config(self):
            return {'clip_value': self.clip_value}      
        
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def wasserstein_loss(self, y_true, y_pred):
        """calculate wasserstein loss"""
        return backend.mean(y_true * y_pred) 
    
    def build_discriminator(self):
        """Override building of the discriminator. In this case we are actually billding the WGAN 'critic'"""
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # weight constraint
        const = self.ClipConstraint(0.01)
        # define model
        model = Sequential()
        # downsample to 14x14
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const, input_shape=self.in_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # downsample to 7x7
        model.add(Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init, kernel_constraint=const))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # scoring, linear activation
        model.add(Flatten())
        model.add(Dense(1))
        # compile model
        self.d_model = model
        opt = RMSprop(lr=0.00005)
        self.d_model.compile(loss=self.wasserstein_loss, optimizer=opt, metrics=['accuracy'])
    
    def build_generator(self):
        """Override building the standalone generator model"""
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # define model
        model = Sequential()
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        model.add(Dense(n_nodes, kernel_initializer=init, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Reshape((7, 7, 128)))
        # upsample to 14x14
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # upsample to 28x28
        model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.2))
        # output 28x28x1
        model.add(Conv2D(1, (7,7), activation='tanh', padding='same', kernel_initializer=init))
        self.g_model = model

    def build_gan(self):
        """override defining of the combined generator and critic model, for updating the generator"""
        critic = self.d_model
        
        # make weights in the critic not trainable
        critic.trainable = False
        # connect them
        model = Sequential()
        # add generator
        model.add(self.g_model)
        # add the critic
        model.add(critic)
        # compile model
        opt = RMSprop(lr=0.00005)
        self.gan_model = model
        self.gan_model.compile(loss=self.wasserstein_loss, optimizer=opt)
    
    def generate_real_samples(self, X, n_samples):
        selected_X, _ = super().generate_real_samples(X, n_samples)
        # For WGAN use target as -1 for real 
        return selected_X, -np.ones((n_samples, 1))

    def generate_generator_prediction_samples(self, n_samples):
        """use the generator to generate n fake examples"""
        selected_X, _ = super().generate_generator_prediction_samples(n_samples)
        # For WGAN use target as 1 for fake
        return selected_X, np.ones((n_samples, 1))

    
    def train(self, X, n_epochs=100, n_batch=128, reporting_period=10, n_critic=5):
        """train the generator and discriminator"""
        bat_per_epo = int(X.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        
        # for recording metrics.
        self.g_loss = np.zeros((n_epochs * bat_per_epo, 1))
        self.d_loss_real = np.zeros((n_epochs * bat_per_epo, 1))        
        self.d_loss_fake = np.zeros((n_epochs * bat_per_epo, 1))
        self.d_acc_real = np.zeros((n_epochs * bat_per_epo, 1))
        self.d_acc_fake = np.zeros((n_epochs * bat_per_epo, 1))
        self.d_acc_real_epochs = np.zeros((n_epochs, 1))
        self.d_acc_fake_epochs = np.zeros((n_epochs, 1))
        
        # manually enumerate epochs
        for i in range(n_epochs):
            
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # update the critic more than the generator
                c1_tmp, c2_tmp = list(), list()    
                for _ in range(n_critic):
                    # update critic model weights on randomly selected 'real' samples
                    X_real, y_real = self.generate_real_samples(X, half_batch)
                    c_loss1, _ = self.d_model.train_on_batch(X_real, y_real)
                    c1_tmp.append(c_loss1)
                    
                    # update discriminator model weights on generated 'fake' examples
                    X_fake, y_fake = self.generate_generator_prediction_samples(half_batch)
                    c_loss2, _ = self.d_model.train_on_batch(X_fake, y_fake)
                    c2_tmp.append(c_loss2)
                # store critic loss
                d_loss1 = np.mean(c1_tmp)
                d_loss2 = np.mean(c2_tmp)
                
                # prepare points in latent space as input for the generator
                z_input = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                y_gan = -np.ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan_model.train_on_batch(z_input, y_gan)
                
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

                # record losses for later
                self.g_loss[i*bat_per_epo + j] = g_loss
                self.d_loss_real[i*bat_per_epo + j] = d_loss1
                self.d_loss_fake[i*bat_per_epo + j] = d_loss2
                self.d_acc_real[i*bat_per_epo + j] = 0
                self.d_acc_fake[i*bat_per_epo + j] = 0

            # save per epoch metrics 
            # evaluate discriminator on real examples
            n_samples = 100
            x_real, y_real = self.generate_real_samples(X, n_samples)
            _, acc_real = self.d_model.evaluate(x_real, y_real, verbose=0)
            self.d_acc_real_epochs[i] = acc_real
            
            # evaluate discriminator on fake examples
            x_fake, y_fake = self.generate_generator_prediction_samples(n_samples)
            _, acc_fake = self.d_model.evaluate(x_fake, y_fake, verbose=0)
            self.d_acc_fake_epochs[i] = acc_fake
            
            # every reporting_period, plot out images.
            if i == 0 or (i+1) % reporting_period == 0 or (i+1) == n_epochs:
                self.summarize_performance(i+1, 'wgan')
                
        # save the generator model
        self.save_model('wgan')