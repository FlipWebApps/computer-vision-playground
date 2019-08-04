import numpy as np
from keras.initializers import RandomNormal
from keras.layers import Activation, Concatenate, Conv2DTranspose, BatchNormalization, Dense, Dropout, Embedding, Input, Flatten, Reshape
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Model
from keras.optimizers import Adam
from matplotlib import pyplot

from models.conditional_gan import ConditionalGAN

class ACGAN(ConditionalGAN):
  
    def build_discriminator(self):
        """Override building of the discriminator."""
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # image input
        in_image = Input(shape=self.in_shape)
        # downsample to 14x14
        fe = Conv2D(32, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(in_image)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # normal
        fe = Conv2D(64, (3,3), padding='same', kernel_initializer=init)(fe)
        fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # downsample to 7x7
        fe = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(fe)
        fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # normal
        fe = Conv2D(256, (3,3), padding='same', kernel_initializer=init)(fe)
        fe = BatchNormalization()(fe)
        fe = LeakyReLU(alpha=0.2)(fe)
        fe = Dropout(0.5)(fe)
        # flatten feature maps
        fe = Flatten()(fe)
        # real/fake output
        out1 = Dense(1, activation='sigmoid')(fe)
        # class label output
        out2 = Dense(self.n_classes, activation='softmax')(fe)
        # define model
        model = Model(in_image, [out1, out2])
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
        return model
    
    def build_generator(self):
        """Override building the standalone generator model"""
        # weight initialization
        init = RandomNormal(stddev=0.02)
        # label input
        in_label = Input(shape=(1,))
        # embedding for categorical input
        li = Embedding(self.n_classes, 50)(in_label)
        # linear multiplication
        n_nodes = 7 * 7
        li = Dense(n_nodes, kernel_initializer=init)(li)
        # reshape to additional channel
        li = Reshape((7, 7, 1))(li)
        # image generator input
        in_lat = Input(shape=(self.latent_dim,))
        # foundation for 7x7 image
        n_nodes = 384 * 7 * 7
        gen = Dense(n_nodes, kernel_initializer=init)(in_lat)
        gen = Activation('relu')(gen)
        gen = Reshape((7, 7, 384))(gen)
        # merge image gen and label input
        merge = Concatenate()([gen, li])
        # upsample to 14x14
        gen = Conv2DTranspose(192, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(merge)
        gen = BatchNormalization()(gen)
        gen = Activation('relu')(gen)
        # upsample to 28x28
        gen = Conv2DTranspose(1, (5,5), strides=(2,2), padding='same', kernel_initializer=init)(gen)
        out_layer = Activation('tanh')(gen)
        # define model
        model = Model([in_lat, in_label], out_layer)
        return model

    def build_gan(self):
        """override defining of the combined generator and discriminator model, for updating the generator"""
        # make weights in the discriminator not trainable
        self.d_model.trainable = False
        # connect the outputs of the generator to the inputs of the discriminator
        gan_output = self.d_model(self.g_model.output)
        # define gan model as taking noise and label and outputting real/fake and label outputs
        model = Model(self.g_model.input, gan_output)
        # compile model
        opt = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
        return model

    def train(self, X, labels, n_epochs=100, n_batch=128, reporting_period=10, n_critic=5):
        """train the generator and discriminator"""
        bat_per_epo = int(X.shape[0] / n_batch)
        half_batch = int(n_batch / 2)
        
        # for recording metrics.
        self.g_loss_authenticity = np.zeros((n_epochs * bat_per_epo, 1))
        self.g_loss_label = np.zeros((n_epochs * bat_per_epo, 1))
        self.d_loss_real_authenticity = np.zeros((n_epochs * bat_per_epo, 1))        
        self.d_loss_real_label = np.zeros((n_epochs * bat_per_epo, 1))        
        self.d_loss_fake_authenticity = np.zeros((n_epochs * bat_per_epo, 1))        
        self.d_loss_fake_label = np.zeros((n_epochs * bat_per_epo, 1))        
        
        # manually enumerate epochs
        for i in range(n_epochs):
            
            # enumerate batches over the training set
            for j in range(bat_per_epo):
                # update discriminator model weights on randomly selected 'real' samples
                [X_real, labels_real], y_real = self.generate_real_samples(X, labels, half_batch)
                _, d_loss_r_1, d_loss_r_2 = \
                    self.d_model.train_on_batch(X_real, [y_real, labels_real])
                
                # update discriminator model weights on generated 'fake' examples
                [X_fake, labels_fake], y_fake = self.generate_generator_prediction_samples(half_batch)
                _, d_loss_f_1, d_loss_f_2 = \
                    self.d_model.train_on_batch(X_fake, [y_fake, labels_fake])
                
                # prepare points in latent space as input for the generator
                [z_input, labels_input] = self.generate_latent_points(n_batch)
                # create inverted labels for the fake samples
                y_gan = np.ones((n_batch, 1))
                # update the generator via the discriminator's error
                _, g_loss_1, g_loss_2 = \
                    self.gan_model.train_on_batch([z_input, labels_input], [y_gan, labels_input])
                
                # summarize loss on this batch
                print('>%d, %d/%d, dr[%.3f,%.3f], df[%.3f,%.3f], g[%.3f,%.3f]' % 
                      (i+1, j+1, bat_per_epo, d_loss_r_1, d_loss_r_2, 
                       d_loss_f_1, d_loss_f_2, 
                       g_loss_1, g_loss_2))

                # record losses for later
                self.g_loss_authenticity[i*bat_per_epo + j] = g_loss_1
                self.g_loss_label[i*bat_per_epo + j] = g_loss_2
                self.d_loss_real_authenticity[i*bat_per_epo + j] = d_loss_r_1
                self.d_loss_real_label[i*bat_per_epo + j] = d_loss_r_2
                self.d_loss_fake_authenticity[i*bat_per_epo + j] = d_loss_f_1
                self.d_loss_fake_label[i*bat_per_epo + j] = d_loss_f_2
            
            # every reporting_period, plot out images.
            if i == 0 or (i+1) % reporting_period == 0 or (i+1) == n_epochs:
                self.summarize_performance(i+1, 'ac-gan')
                
        # save the generator model
        self.save_model('ac-gan')
    
    def plot_training_metrics(self, figsize = (10,10)):
        fig = pyplot.figure(figsize=figsize)
        
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.plot(self.d_loss_real_authenticity, label='Discriminator Loss Authenticity (real)')
        pyplot.plot(self.d_loss_real_label, label='Discriminator Loss Label (real)')
        pyplot.plot(self.d_loss_fake_authenticity, label='Discriminator Loss Authenticity (fake)')
        pyplot.plot(self.d_loss_fake_label, label='Discriminator Loss Label (fake)')
        pyplot.plot(self.g_loss_authenticity, label='Generator Loss Authenticity')
        pyplot.plot(self.g_loss_label, label='Generator Loss Label')
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