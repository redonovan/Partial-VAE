# Partial VAE - implementation.
# 4-6 Feb 2022 (v1).


import time
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib
import matplotlib.pyplot as plt

# GPU memory hack
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Hyperparameters
# model
lat_dim       = 20       # free choice, latent dimension, 20 in (Ma et al, 2019)
elh_dim       = 500      # free choice, encoder/decoder large hidden layer dimension, 500 in (Ma et al, 2019)
esh_dim       = 200      # free choice, encoder/decoder small hidden layer dimension, 200 in (Ma et al, 2019)
pne_dim       = 20       # free choice, pointnet embedding dimension, 20 in (Ma et al, 2019)
pnc_dim       = 500      # free choice, pointnet feature vector "c" dimension, 500 in (Ma et al, 2019)

# training
batch_size    = 100      # free choice, 100 in (Ma et al, 2019)
max_mrp       = 0.7      # free choice, max missing rate parameter, 0.7 in (Ma et al, 2019)
learning_rate = 0.001    # free choice, for Adam, 0.001 in (Ma et al, 2019)
num_epochs    = 100      # free choice, 5 deduced from 3K iterations in (Ma et al, 2019)
kl_weight     = 1.0/1000 # Kullback-Leibler weight used in ELBO calculations


# Data

(x_train, _), (x_test, y_test) = mnist.load_data()

x_train  = x_train.astype('float32') / 255.0
x_train  = x_train.reshape(x_train.shape[0],-1)
x_test   = x_test.astype('float32') / 255.0
x_test   = x_test.reshape(x_test.shape[0],-1)

train_ds = tf.data.Dataset.from_tensor_slices(x_train)
train_ds = train_ds.batch(batch_size)
val_ds   = tf.data.Dataset.from_tensor_slices(x_test)
val_ds   = val_ds.batch(batch_size)

# image dimension in pixels
img_dim = x_train.shape[1]




# Model

# The encoder converts a pointnet vector into the mean u and log variance lv of a normal distn over latent
# dimensions.  That is, the posterior distribution q(z|x) = N(z; u, exp(lv) o I) where o is element-wise
# multiplication.  Log variance is predicted to ensure that variance is always positive.

class Encoder(layers.Layer):
    '''Following (Ma et al, 2019)'''
    def __init__(self, lat_dim, elh_dim, esh_dim, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        # store parameters
        self.lat_dim = lat_dim
        self.elh_dim = elh_dim
        self.esh_dim = esh_dim
        # input to hidden layer 1
        self.ih1  = layers.Dense(elh_dim, activation='relu')
        # hidden layer 1 to hidden layer 2
        self.h1h2 = layers.Dense(elh_dim, activation='relu')
        # hidden layer 2 to hidden layer 3
        self.h2h3 = layers.Dense(esh_dim, activation='relu')
        # hidden layer 3 to latent means
        self.h3u  = layers.Dense(lat_dim)
        # hidden layer 3 to latent log variances
        self.h3lv = layers.Dense(lat_dim)
        #
    def call(self, inputs):
        # inputs are pointnet vectors, (batch, pnc_dim)
        # compute hidden layer 1,      (batch, elh_dim)
        h1 = self.ih1(inputs)
        # compute hidden layer 2,      (batch, elh_dim)
        h2 = self.h1h2(h1)
        # compute hidden layer 3,      (batch, esh_dim)
        h3 = self.h2h3(h2)
        # compute latent means,        (batch, lat_dim)
        u  = self.h3u(h3)
        # compute latent log vars,     (batch, lat_dim)
        lv = self.h3lv(h3)
        # return latent means, log vars
        return u, lv



# The decoder converts latent vectors z into Bernoulli distribution parameters, one for every pixel
# in the image.

class Decoder(layers.Layer):
    '''Following (Ma et al, 2019)'''
    def __init__(self, img_dim, elh_dim, esh_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        # store parameters
        self.img_dim = img_dim
        self.elh_dim = elh_dim
        self.esh_dim = esh_dim
        # latent to hidden layer 1
        self.lh1  = layers.Dense(esh_dim, activation='relu')
        # hidden layer 1 to hidden layer 2
        self.h1h2 = layers.Dense(elh_dim, activation='relu')
        # hidden layer 2 to hidden layer 3
        self.h2h3 = layers.Dense(elh_dim, activation='relu')
        # hidden layer 3 to Bernoulli distribution parameters
        self.h3y  = layers.Dense(img_dim, activation='sigmoid')
        #
    def call(self, inputs):
        # inputs (latent vectors), (batch, lat_dim)
        # compute hidden layer 1,  (batch, esh_dim)
        h1 = self.lh1(inputs)
        # compute hidden layer 2,  (batch, elh_dim)
        h2 = self.h1h2(h1)
        # compute hidden layer 3,  (batch, elh_dim)
        h3 = self.h2h3(h2)
        # compute Bernoulli parms, (batch, img_dim)
        y  = self.h3y(h3)
        # return Bernoulli parms
        return y



# The Partial VAE model includes both encoder and decoder, and a Pointnet feature vector function.
# The model is trained by maximizing the Variational Lower Bound on observed pixels by encoding
# training examples, via the Pointnet function and the encoder, into latent variable distributions,
# sampling from those distributions, and decoding the samples into predicted images, which, since
# this is an autoencoder, should be similar to the observed pixels of the training examples.

class PartialVAE(keras.Model):
    def __init__(self, img_dim, lat_dim, elh_dim, esh_dim, pne_dim, pnc_dim, **kwargs):
        super(PartialVAE, self).__init__(**kwargs)
        # store parameters
        self.img_dim = img_dim # image dimension
        self.lat_dim = lat_dim # latent dimension
        self.elh_dim = elh_dim # encoder/decoder large hidden dimension
        self.esh_dim = esh_dim # encoder/decoder small hidden dimension
        self.pne_dim = pne_dim # pointnet embedding dimension
        self.pnc_dim = pnc_dim # pointnet feature vector "c" dimension
        # encoder & decoder
        self.encoder = Encoder(lat_dim, elh_dim, esh_dim)
        self.decoder = Decoder(img_dim, elh_dim, esh_dim)
        # pointnet
        self.embed   = self.add_weight(shape=(self.img_dim, self.pne_dim),
                                       initializer="random_normal",
                                       name="embedding_matrix",
                                       trainable=True)
        self.pnnn    = layers.Dense(pnc_dim, activation='relu')
        #
    def compute_ELBOs(self, inputs, mask, y, u, lv):
        # The ELBO is computed from log(p_model(x|z)) where x is drawn from the data distribution,
        # minus a Kullback Leibler term which will be dealt with below.  The expectation of the
        # -ve log of p_model(x) is the cross-entropy loss, see Deep Learning (Goodfellow, et al.)
        # p 129, in this case computed from binary_crossentropy(true, pred).  Since the Encoder
        # did not see masked input pixels, one cross-entropy is computed between observed inputs
        # and predictions, and another between unobserved inputs and predictions as an imputation
        # performance measure.
        # 
        # compute observed inputs and observed predictions by zeroing out the unobserved values
        oi   = tf.math.multiply(inputs, tf.cast(mask, tf.float32))                 # (batch, img_dim)
        oy   = tf.math.multiply(y     , tf.cast(mask, tf.float32))                 # (batch, img_dim)
        # compute the observed cross entropy loss
        # note that unobserved pixels will not contribute because 0ln0 + 1ln1 = 0 + 0 = 0
        oxel = tf.keras.losses.binary_crossentropy(oi, oy)                         # (batch, )
        # 
        # compute the same for unobserved values
        ui   = tf.math.multiply(inputs, tf.cast(tf.logical_not(mask), tf.float32)) # (batch, img_dim)
        uy   = tf.math.multiply(y     , tf.cast(tf.logical_not(mask), tf.float32)) # (batch, img_dim)
        uxel = tf.keras.losses.binary_crossentropy(ui, uy)                         # (batch, )
        #
        # The other term in the ELBO is the Kullback Leibler divergence DKL(q(z|x)||p(z)).
        # This is given, see Kingma and Welling, 2014, by
        DKL  = -0.5 * tf.reduce_sum(1 + lv - tf.square(u) - tf.exp(lv), axis=-1)   # (batch, )
        #
        # The ELBOs are the -ve xent minus the (weighted) DKL term
        oELBO = -oxel - kl_weight * DKL                                            # (batch, )
        uELBO = -uxel - kl_weight * DKL                                            # (batch, )
        # Both the ELBOs have shape (batch, ) so take the mean over the batch here   ()
        return tf.reduce_mean(oELBO), tf.reduce_mean(uELBO)
    #
    def sample(self, u, lv):
        # first sample epsilon from N(0, I),                        (batch, lat_dim)
        epsilon = tf.random.normal([tf.shape(u)[0], self.lat_dim])
        # shift and scale epsilon to sample from N(u, exp(lv) o I), (batch, lat_dim)
        return u + epsilon * tf.exp(0.5 * lv)
    #
    def pointnet(self, inputs, mask):
        # This function implements (Ma et al, 2019)'s Pointnet-plus encoding.
        # inputs are images,    (batch, img_dim)
        # mask is a mask,       (batch, img_dim) False => masked
        #
        # Under the PNP model we multiply the pixel embeddings by the input values
        emi = tf.math.multiply(inputs[:,:,None], self.embed)             # (batch, img_dim, pne_dim)
        # Now apply the pointnet neural network "h"
        pnh = self.pnnn(emi)                                             # (batch, img_dim, pnc_dim)
        # Now apply the mask, zeroing out masked pixel feature vectors
        gin = tf.math.multiply(pnh, tf.cast(mask, tf.float32)[:,:,None]) # (batch, img_dim, pnc_dim)
        # Finally apply operation "g", a simple sum over pixels, knowing that masked pixels do not contribute
        pnc = tf.reduce_sum(gin, axis=1)                                 # (batch, pnc_dim)
        # Return pnc, the pointnet-plus feature vectors "c" for the inputs
        return pnc
    #
    def call(self, inputs, mask):
        # inputs are images,    (batch, img_dim)
        # mask is boolean,      (batch, img_dim) where False means masked
        #
        # turn the images into pointnet feature vectors "c"
        pnc = self.pointnet(inputs, mask)     # (batch, pnc_dim)
        # encode each pointnet vector into a normal distribution over latent dimensions N(u, exp(lv) o I)
        # u and lv shapes                       (batch, lat_dim)
        u, lv = self.encoder(pnc)
        # sample from each normal distribution, (batch, lat_dim)
        z     = self.sample(u, lv)
        # decode an image from each sampled z,  (batch, img_dim)
        y     = self.decoder(z)
        # compute the observed and unobserved ELBOs
        oELBO, uELBO  = self.compute_ELBOs(inputs, mask, y, u, lv) # ()
        # the VAE loss for training purposes is the -ve observed ELBO
        self.add_loss(-oELBO)
        # return the unobserved ELBO, and u and lv (which may be used to plot images)
        return uELBO, u, lv


# instantiate the model
vae = PartialVAE(img_dim, lat_dim, elh_dim, esh_dim, pne_dim, pnc_dim)

# define optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# define accumulators
train_loss  = tf.keras.metrics.Mean()
val_loss    = tf.keras.metrics.Mean()
val_uELBO   = tf.keras.metrics.Mean()

# setup tensorboard logging directories
# specify -p 6010:6010 in docker run command for tensorflow image
# run "tensorboard --logdir logs --host 0.0.0.0 --port 6010" in docker shell
# point browser at http://localhost:6010/

current_time         = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir        = 'logs/' + current_time + '/train'
val_log_dir          = 'logs/' + current_time + '/val'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer   = tf.summary.create_file_writer(val_log_dir)


@tf.autograph.experimental.do_not_convert
def generate_random_mask(sh, missing_rate_parameter):
    # produce a float tensor sampled from U(0, 1.0) with the specified shape
    ft   = tf.random.uniform(sh, maxval=1.0, dtype=tf.float32)
    # produce a mask tensor in which False means masked and True means use-this-value
    mask = ft > missing_rate_parameter
    return mask


# train step
signature = [tf.TensorSpec(shape=(None, img_dim), dtype=tf.float32)]

@tf.function(input_signature = signature)
@tf.autograph.experimental.do_not_convert
def train_step(d):
    #
    # in training sample from U(0, 0.7) to produce a missing rate parameter
    mrp  = tf.random.uniform((), maxval=max_mrp)         # ()
    # generate random masks for the images in d
    mask = generate_random_mask(tf.shape(d), mrp)        # (batch, img_dim)
    #
    with tf.GradientTape() as tape:
        # call the model, computing the training set loss
        _, _, _ = vae(d, mask)
        # obtain the loss
        loss = sum(vae.losses)
    #
    # compute and apply the gradients
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
    # accumulate the training set loss
    train_loss(loss)


# val step
@tf.function(input_signature = signature)
@tf.autograph.experimental.do_not_convert
def val_step(d):
    # on validation data the missing rate parameter is fixed at 0.7, per (Ma et al, 2019)
    mrp  = max_mrp                                       # ()
    # generate random masks for the images in d
    mask = generate_random_mask(tf.shape(d), mrp)        # (batch, img_dim)
    # call the model
    uELBO, _, _ = vae(d, mask)
    # obtain the loss
    loss  = sum(vae.losses)
    # accumulate the validation set loss and (missing pixel) uELBO
    val_loss(loss)
    val_uELBO(uELBO)



# training loop
for epoch in range(num_epochs):
    start = time.time()
    #
    # reset accumulators
    train_loss.reset_states()
    val_loss.reset_states()
    val_uELBO.reset_states()
    #
    for d in train_ds:
        train_step(d)
    #
    print(f'epoch {epoch+1:3d} train loss {train_loss.result():.4f}, ', end='')
    # tensorboard log
    with train_summary_writer.as_default():
        _ = tf.summary.scalar('loss', train_loss.result(), step=epoch)
    # validate epoch
    for d in val_ds:
        val_step(d)
    #
    print(f'val loss {val_loss.result():.4f}, ', end='')
    print(f'val uELBO {val_uELBO.result():.4f}, ', end='')
    # tensorboard log
    with val_summary_writer.as_default():
        _ = tf.summary.scalar('loss', val_loss.result(), step=epoch)
    #
    print(f'time taken {time.time() - start:.2f}s')
    

# Use the model generatively to produce images from the prior N(0,I).
# Produce a 15x15 grid of images from latent vectors sampled randomly from N(0,I).

n = 15
digit_size = int(np.sqrt(img_dim))
figure = np.zeros((digit_size * n, digit_size * n))

for i in range(n):
    for j in range(n):
        z_sample = tf.random.normal((lat_dim,))
        z_image  = vae.decoder(z_sample[None,:])
        digit    = tf.reshape(z_image, [digit_size, digit_size])
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10,10))
plt.imshow(figure, cmap='Greys_r')
plt.show()



# Choose a test-set image number to obscure, and plot it.
tin   = 4

image = x_test[tin]
digit = tf.reshape(image, [digit_size, digit_size])
plt.figure(figsize=(5,5))
plt.imshow(digit, cmap='Greys_r')
plt.show()

@tf.autograph.experimental.do_not_convert
def generate_continuous_mask(sh, lo, hi):
    # Generate a mask with shape sh which is False over [lo:hi) and True everywhere else.
    # Note that False means masked and True means use-this-value.
    mask = np.ones(sh, dtype=bool)
    mask[lo:hi] = False
    mask = tf.convert_to_tensor(mask)
    return mask


# generate a mask to obscure the first 60% of the test image
lo   = 0
hi   = int(0.6 * img_dim)
mask = generate_continuous_mask(img_dim, lo, hi)

# obscure the test image and plot it
obsim = tf.math.multiply(image, tf.cast(mask, tf.float32))
digit = tf.reshape(obsim, [digit_size, digit_size])
plt.figure(figsize=(5,5))
plt.imshow(digit, cmap='Greys_r')
plt.show()

# Call the model with the obscured image and its obscuring mask.
# This returns the latent space posterior distribution of the obscured image, q(z|x).
_, u, lv = vae(obsim[None,:], mask[None,:])

# plot images generated from random samples from q(z|x)

n = 15
digit_size = int(np.sqrt(img_dim))
figure = np.zeros((digit_size * n, digit_size * n))

for i in range(n):
    for j in range(n):
        z_sample = vae.sample(u, lv)
        z_image  = vae.decoder(z_sample)
        digit    = tf.reshape(z_image, [digit_size, digit_size])
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10,10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

