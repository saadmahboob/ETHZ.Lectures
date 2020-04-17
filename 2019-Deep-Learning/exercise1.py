import tensorflow as tf
import numpy as np


# Dimensions and desired intermediate sizes (here uniform)
d = 30
n = 3
sizes = [i * n for i in range(1,d//n)]
assert(d % n == 0)


def compress(inputs, sizes):
  '''
    inputs: [1, d] shaped tensor
    sizes: list of dimensions to reduce inputs (iteratively) to
  '''
  last_d = inputs.get_shape()[1]
  for d in sizes:
    W = tf.get_variable("W_%d_%d" % (last_d, d), [last_d, d], tf.float32, initializer=tf.random_uniform_initializer())
    print([last_d, d])
    inputs = tf.matmul(inputs, W) 
    last_d = d
    
  return inputs

def decompress(inputs, sizes):
  '''
    inputs: [1, d] shaped tensor
    sizes: list of dimensions to expand inputs (iteratively) to
  '''
  last_d = inputs.get_shape()[1]
  for d in sizes[1:]:
    W = tf.get_variable("W_%d_%d" % (d, last_d), [d, last_d], tf.float32, initializer=tf.random_uniform_initializer())
    inputs = tf.matmul(inputs, W, transpose_b=True)
    last_d = d
    
  return inputs

# Create graph
x = tf.placeholder(tf.float32, [1, d], name="input")

with tf.variable_scope("autoencoder") as scope:
  x_prime = compress(x, sizes[::-1])
  # Allow reuse of matrices
  scope.reuse_variables()
  x_tilde = decompress(x_prime, sizes)


# Start session to run the graph
with tf.Session() as session:
  # Initialize & finalize graph
  session.run(tf.global_variables_initializer())
  session.graph.finalize()
  
  for inputs in np.random.random_sample([10, d]):
    # inputs is of shape [d], so we add an outer dimension to match the palceholder
    feed_dict = {x : [inputs]}
    outputs = session.run(x_tilde, feed_dict=feed_dict)
    print(outputs)