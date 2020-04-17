# coding: utf-8

# # Loading libraries
import tensorflow as tf
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
sess = tf.InteractiveSession()

tf.set_random_seed(0)
np.random.seed(0)
rng = np.random.RandomState()

## Importing mnist dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = np.reshape(x_train, [-1, 28*28])
x_test = np.reshape(x_test, [-1, 28*28])
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

## Validation data split
x_train, x_val = x_train[:55000], x_train[55000:]
y_train, y_val = y_train[:55000], y_train[55000:]

## Network Parameters
n_input        = ...
n_hidden_1     = ...
n_hidden_2     = ...
n_hidden_3     = ...

## Training Parameters
learning_rate   = 0.001
training_epochs = 20
batch_size      = 100
display_step    = 1

## Xavier Initialization
def xavier_init(dim_in, dim_out, constant=4):
    ...

## Initializing weights & biases
weights = {
    'encoder_h1': ...,
    'encoder_h2': ...,
    'decoder_h1': ...,
    'decoder_h2': ...
}
biases = {
    'encoder_b1': ...,
    'encoder_b2': ...,
    'decoder_b1': ...,
    'decoder_b2': ...
}

## Creating Encoder
def encoder(x, activ_fnc=tf.nn.sigmoid):
    ...


## Creating Decoder
def decoder(x, activ_fnc=tf.nn.sigmoid):
    ...


## Initializing Model
x_plh = tf.placeholder("float", [None, n_input])
enc_model = encoder(x_plh)
dec_model = decoder(enc_model)
y_pred = dec_model
y_gt   = x_plh

## Defining loss-function & optimizer
x_error   = ...
sq_error  = ...
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(...)

## Initializing the variables
init = tf.global_variables_initializer()
sess.run(init)

def batch_indices(batch_nb, data_length, batch_size):
  # Batch start and end index
  start = int(batch_nb * batch_size)
  end = int((batch_nb + 1) * batch_size)
  # When there are not enough inputs left, we reuse some to complete the batch
  if end > data_length:
    shift = end - data_length
    start -= shift
    end -= shift
  return start, end

## Training
train_errors = []
valid_errors = []
for epoch in range(training_epochs):
    nb_batches = int(math.ceil(float(len(x_train)) / batch_size))
    assert nb_batches * batch_size >= len(x_train)

    # Indices to re-shuffle training set for every epoch
    index_shuf = list(range(len(x_train)))
    rng.shuffle(index_shuf)

    # Loop over all batches
    for batch in range(nb_batches):
        start, end = batch_indices(batch, len(x_train), batch_size)
        sess.run(optimizer, feed_dict={x_plh: x_train[index_shuf[start:end]]})
      
    train_loss = sess.run(x_error, feed_dict={x_plh: x_train})
    validation_loss = sess.run(x_error, feed_dict={x_plh: x_val})
    train_errors.append(train_loss)
    valid_errors.append(validation_loss)
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), ", training loss=", "{:.7f}".format(train_loss), ", validation loss=",
              "{:.7f}".format(validation_loss))

print("Training finished!")
plt.plot(valid_errors)
plt.xlabel("# epochs")
plt.ylabel("Loss")

## Testing
test_loss = sess.run(x_error, feed_dict={x_plh: x_test})
print("Test loss: ", "{:.7f}".format(test_loss))

## Visualization
noSamples=5
xsample = x_test[:noSamples]
xreconst = sess.run(y_pred, feed_dict={x_plh: xsample})

fig = plt.figure(figsize=(5, noSamples*2))
for ii in range(noSamples):
    plt.subplot(noSamples, 2, 2*ii + 1)
    plt.title("Input")
    plt.imshow(np.reshape(xsample[ii], (28,28)), vmin=0, vmax=1)
    plt.colorbar()
    
    plt.subplot(noSamples, 2, 2*ii + 2)
    plt.title("Reconstruction")
    plt.imshow(np.reshape(xreconst[ii], (28,28)), vmin=0, vmax=1)
    plt.colorbar()

plt.tight_layout()
fig.savefig('out_crossError_xavierInit.png', bbox_inches='tight')
