import numpy as np

from keras.layers import Concatenate, Input, Dense, Reshape, Flatten, Dropout, multiply, \
    BatchNormalization, Activation, Embedding, ZeroPadding2D, Conv2DTranspose
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from matplotlib import pyplot

from models.gan import GAN

class ConditionalGAN(GAN):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
       
    def build_discriminator(self):
        """define the standalone discriminator model"""
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(self.n_classes, 50)(in_label)
        # scale up to image dimensions with linear activation
        n_nodes = self.in_shape[0] * self.in_shape[1]
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((self.in_shape[0], self.in_shape[1], 1))(li)
        # image input
        in_image = Input(shape=self.in_shape)
        # concat label as a channel
        merge = Concatenate()([in_image, li])
        # downsample
        fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
        fe = LeakyReLU(alpha=0.2)(fe)
        # downsample
        fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        # flatten feature maps
        fe = Flatten()(fe)
        # dropout
        fe = Dropout(0.4)(fe)
        # output
        out_layer = Dense(1, activation='sigmoid')(fe)
        # define model
        model = Model([in_image, in_label], out_layer)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return model
    
    def build_generator(self):
        """define the standalone generator model"""
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(self.n_classes, 50)(in_label)
        # linear multiplication
        n_nodes = 7 * 7
        li = Dense(n_nodes)(li)
        # reshape to additional channel
        li = Reshape((7, 7, 1))(li)
        # image generator input
        in_lat = Input(shape=(self.latent_dim,))
        # foundation for 7x7 image
        n_nodes = 128 * 7 * 7
        gen = Dense(n_nodes)(in_lat)
        gen = LeakyReLU(alpha=0.2)(gen)
        gen = Reshape((7, 7, 128))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        # upsample to 14x14
        gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
        gen = LeakyReLU(alpha=0.2)(gen)
        # upsample to 28x28
        gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
        gen = LeakyReLU(alpha=0.2)(gen)
        # output
        out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
        # define model
        model = Model([in_lat, in_label], out_layer)
        return model

    def build_gan(self):
        """define the combined generator and discriminator model, for updating the generator"""
        # make weights in the discriminator not trainable
        self.d_model.trainable = False
        # get noise and label inputs from generator model
        gen_noise, gen_label = self.g_model.input
        # get image output from the generator model
        gen_output = self.g_model.output
        # connect image output and label input from generator as inputs to discriminator
        gan_output = self.d_model([gen_output, gen_label])
        # define gan model as taking noise and label and outputting a classification
        model = Model([gen_noise, gen_label], gan_output)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model  
    
    def generate_real_samples(self, X, labels, n_samples):
        """select real samples"""
        # choose random instances
        ix = np.random.randint(0, X.shape[0], n_samples)
        # select images and labels
        selected_X, selected_labels = X[ix], labels[ix]
        # generate 'real' class labels (1)
        y = np.ones((n_samples, 1))
        return [selected_X, selected_labels], y

    def generate_latent_points(self, n_samples):
        """generate points in latent space as input for the generator"""
        # generate points in the latent space
        latent_points = super().generate_latent_points(n_samples)
        # generate labels
        labels = np.random.randint(0, self.n_classes, n_samples)
        return [latent_points, labels]

    def generate_generator_prediction_samples(self, n_samples):
        """use the generator to generate n fake examples"""
        # generate points in latent space
        latent_points, labels = self.generate_latent_points(n_samples)
        # predict outputs
        X = self.generator_prediction(latent_points, labels)
        # create class labels
        y = np.zeros((n_samples, 1))
        return [X, labels], y
      
    def generator_prediction(self, latent_points, labels):
        return self.g_model.predict([latent_points, labels])
    
    def train(self, X, labels, n_epochs=100, n_batch=128, reporting_period=10):
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
                # update discriminator model weights on randomly selected 'real' samples
                [X_real, labels_real], y_real = self.generate_real_samples(X, labels, half_batch)
                d_loss1, d_acc1  = self.d_model.train_on_batch([X_real, labels_real], y_real)
                
                # update discriminator model weights on generated 'fake' examples
                [X_fake, labels_fake], y_fake = self.generate_generator_prediction_samples(half_batch)
                d_loss2, d_acc2 = self.d_model.train_on_batch([X_fake, labels_fake], y_fake)
                
                # prepare points in latent space as input for the generator
                [z_input, labels_input] = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((n_batch, 1))
                # update the generator via the discriminator's error
                g_loss = self.gan_model.train_on_batch([z_input, labels_input], y_gan)
                
                # summarize loss on this batch
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                    (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

                # record losses for later
                self.g_loss[i*bat_per_epo + j] = g_loss
                self.d_loss_real[i*bat_per_epo + j] = d_loss1
                self.d_loss_fake[i*bat_per_epo + j] = d_loss2
                self.d_acc_real[i*bat_per_epo + j] = d_acc1
                self.d_acc_fake[i*bat_per_epo + j] = d_acc2

            # save per epoch metrics 
            # evaluate discriminator on real examples
            n_samples = 100
            x_real, y_real = self.generate_real_samples(X, labels, n_samples)
            _, acc_real = self.d_model.evaluate(x_real, y_real, verbose=0)
            self.d_acc_real_epochs[i] = acc_real
            
            # evaluate discriminator on fake examples
            x_fake, y_fake = self.generate_generator_prediction_samples(n_samples)
            _, acc_fake = self.d_model.evaluate(x_fake, y_fake, verbose=0)
            self.d_acc_fake_epochs[i] = acc_fake
            
            # every reporting_period, plot out images.
            if i == 0 or (i+1) % reporting_period == 0 or (i+1) == n_epochs:
                self.summarize_performance(i+1, 'conditional-gan')
                
        # save the generator model
        self.save_model('conditional-gan')

    def plot_random_generated_images(self):
        """create a plot of randomly generated images (reversed grayscale)"""
        dimensions=(10, 10)
        figsize=(10, 10)
        n_samples=100
        
        (X, _), _ = self.generate_generator_prediction_samples(n_samples)
            
        self.grid_plot(X, dimensions=dimensions, figsize=figsize)        