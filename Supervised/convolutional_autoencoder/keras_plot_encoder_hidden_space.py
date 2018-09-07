from keras.layers import Input, Dense, Lambda, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import fashion_mnist
from keras.datasets import mnist
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

# build a model to project inputs on the latent space
batch_size = 128
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 100
epsilon_std = 1.0

################################################
x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

################################################
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# to reuse these later
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')

# train the VAE on fashion MNIST images
#(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

history = vae.fit(x_train,shuffle=True,epochs=epochs,batch_size=batch_size, validation_data=(x_test, None))

################################################
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
def plot_latentSpace(encoder, x_test, y_test, batch_size):
    x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(6, 6))
    plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test, cmap='tab10')
    plt.colorbar()
    plt.show()


plot_latentSpace(encoder, x_test, y_test, batch_size)
print(len(x_test[0]))














# size of bottleneck latent space
#encoding_dim = 32
# input placeholder
#input_img = Input(shape=(784,))
# encoded representation
#encoded = Dense(encoding_dim, activation='relu')(input_img)
# lossy reconstruction
#decoded = Dense(784, activation='sigmoid')(encoded)
# full AE model: map an input to its reconstruction
#autoencoder = Model(input_img, decoded)

# encoder: map an input to its encoded representation
#encoder = Model(input_img, encoded)
# placeholder for an encoded input
#encoded_input = Input(shape=(encoding_dim,))
# last layer of the autoencoder model
#decoder_layer = autoencoder.layers[-1]
# decoder
#decoder = Model(encoded_input, decoder_layer(encoded_input))
