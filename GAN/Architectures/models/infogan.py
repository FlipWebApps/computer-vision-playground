import numpy as np
from keras import backend
from keras.constraints import Constraint
from keras.initializers import RandomNormal
from keras.layers import Activation, Conv2DTranspose, BatchNormalization, Dense, Flatten, Input, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from models.gan import GAN

class InfoGAN(GAN):
       
    def __init__(self, n_cat=10, *args, **kwargs):
        """Initialisation
        
        Args:
            n_cat (int, optional): Number of control variables to use. Defaults to 10.
        """
        self.n_cat = n_cat
        
        super().__init__(*args, **kwargs)

    def build_discriminator(self):
        """Override building of the discriminator. 
        
        We add an auxilaty model q for predicting the value of the categorical variable. This isn't
        compiled as it is never used in a standalone manner, only as part of the final gan."""
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        in_image = Input(shape=self.in_shape)
        # downsample to 14x14
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
        d = LeakyReLU(alpha=0.1)(d)
        # downsample to 7x7
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.1)(d)
        d = BatchNormalization()(d)
        # normal
        d = Conv2D(256, (4,4), padding='same', kernel_initializer=init)(d)
        d = LeakyReLU(alpha=0.1)(d)
        d = BatchNormalization()(d)
        # flatten feature maps
        d = Flatten()(d)
        # real/fake output
        out_classifier = Dense(1, activation='sigmoid')(d)
        # define d model
        self.d_model = Model(in_image, out_classifier)
        # compile d model
        self.d_model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5), metrics=['accuracy'])

        # create q model layers
        q = Dense(128)(d)
        q = BatchNormalization()(q)
        q = LeakyReLU(alpha=0.1)(q)
        # q model output
        out_codes = Dense(self.n_cat, activation='softmax')(q)
        # define q model
        self.q_model = Model(in_image, out_codes)
    
    def build_generator(self):
        """Override building the standalone generator model.
        
        This will take latent space like a conditional gan and the n_cat control variables."""
        gen_input_size = self.latent_dim + self.n_cat

        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image generator input
        in_lat = Input(shape=(gen_input_size,))
        # foundation for 7x7 image
        n_nodes = 512 * 7 * 7
        gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
        gen = Activation('relu')(gen)
        gen = BatchNormalization()(gen)
        gen = Reshape((7, 7, 512))(gen)
        # normal
        gen = Conv2D(128, (4,4), padding='same', kernel_initializer=init)(gen)
        gen = Activation('relu')(gen)
        gen = BatchNormalization()(gen)
        # upsample to 14x14
        gen = Conv2DTranspose(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
        gen = Activation('relu')(gen)
        gen = BatchNormalization()(gen)
        # upsample to 28x28
        gen = Conv2DTranspose(1, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(gen)
        # tanh output
        out_layer = Activation('tanh')(gen)
        # define model
        self.g_model = Model(in_lat, out_layer)

    def build_gan(self):
        """override defining of the combined generator and discriminator model, for updating the generator"""
        # make weights in the discriminator (some shared with the q model) as not trainable
        self.d_model.trainable = False
        # connect g outputs to d inputs
        d_output = self.d_model(self.g_model.output)
        # connect g outputs to q inputs
        q_output = self.q_model(self.g_model.output)
        # define composite model
        self.gan_model = Model(self.g_model.input, [d_output, q_output])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        self.gan_model.compile(loss=['binary_crossentropy', 'categorical_crossentropy'], optimizer=opt)
    
    def generate_latent_points(self, n_samples):
        """generate points in latent space as input for the generator. we concatonate with control codes"""
        # generate points in the latent space
        latent_points = super().generate_latent_points(n_samples)
        # reshape into a batch of inputs for the network
        latent_points = latent_points.reshape(n_samples, self.latent_dim)

        # generate categorical codes
        cat_codes = np.random.randint(0, self.n_cat, n_samples)
        # one hot encode
        cat_codes = to_categorical(cat_codes, num_classes=self.n_cat)
        # concatenate latent points and control codes
        z_input = np.hstack((latent_points, cat_codes))
        return [z_input, cat_codes]

    def generate_generator_prediction_samples(self, n_samples):
        """use the generator to generate n fake examples"""
        # generate points in latent space and control codes
        z_input, _ = self.generate_latent_points(n_samples)
        # predict outputs
        images = self.generator_prediction(z_input)
        # create class labels
        y = np.zeros((n_samples, 1))
        return images, y
    
    def train(self, X, n_epochs=100, n_batch=128, reporting_period=10, n_critic=5):
        """train the generator and discriminator"""
        bat_per_epo = int(X.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        
        # for recording metrics.

        self.step_metrics = {
            'Discriminator Loss (real)': np.zeros((n_epochs * bat_per_epo, 1)), 
            'Discriminator Loss (fake)': np.zeros((n_epochs * bat_per_epo, 1)), 
            'Discriminator Accuracy (real)': np.zeros((n_epochs * bat_per_epo, 1)), 
            'Discriminator Accuracy (fake)': np.zeros((n_epochs * bat_per_epo, 1)), 
            'GAN Loss': np.zeros((n_epochs * bat_per_epo, 1)),
            'GAN Loss (auxiliary )': np.zeros((n_epochs * bat_per_epo, 1)),
        }        
        self.epoch_metrics = {
            'Discriminator Accuracy (real)': np.zeros((n_epochs, 1)),
            'Discriminator Accuracy (fake)': np.zeros((n_epochs, 1)),
        }   
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

                # update critic model weights on randomly selected 'real' samples
                X_real, y_real = self.generate_real_samples(X, half_batch)
                d_loss1, d_real_accuracy = self.d_model.train_on_batch(X_real, y_real)
                
                # update discriminator model weights on generated 'fake' examples
                X_fake, y_fake = self.generate_generator_prediction_samples(half_batch)
                d_loss2, d_fake_accuracy = self.d_model.train_on_batch(X_fake, y_fake)
                
                # prepare points in latent space as input for the generator
                z_input, cat_codes = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((n_batch, 1))
                # update the generator via the discriminator's error
                _, g_loss1, g_loss2 = self.gan_model.train_on_batch(z_input, [y_gan, cat_codes])
                
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss1))

                # record losses for later
                self.step_metrics['Discriminator Loss (real)'][i*bat_per_epo + j] = d_loss1
                self.step_metrics['Discriminator Loss (fake)'][i*bat_per_epo + j] = d_loss2
                self.step_metrics['Discriminator Accuracy (real)'][i*bat_per_epo + j] = d_real_accuracy
                self.step_metrics['Discriminator Accuracy (fake)'][i*bat_per_epo + j] = d_fake_accuracy
                self.step_metrics['GAN Loss'][i*bat_per_epo + j] = g_loss1
                self.step_metrics['GAN Loss (auxiliary )'][i*bat_per_epo + j] = g_loss2

            # save per epoch metrics 
            # evaluate discriminator on real examples
            n_samples = 100
            x_real, y_real = self.generate_real_samples(X, n_samples)
            _, acc_real = self.d_model.evaluate(x_real, y_real, verbose=0)
            self.epoch_metrics['Discriminator Accuracy (real)'][i] = acc_real
            
            # evaluate discriminator on fake examples
            x_fake, y_fake = self.generate_generator_prediction_samples(n_samples)
            _, acc_fake = self.d_model.evaluate(x_fake, y_fake, verbose=0)
            self.epoch_metrics['Discriminator Accuracy (fake)'][i] = acc_fake
            
            # every reporting_period, plot out images.
            if i == 0 or (i+1) % reporting_period == 0 or (i+1) == n_epochs:
                self.summarize_performance(i+1, 'infogan')
                
        # save the generator model
        self.save_model('infogan')
