"""Model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


NUM_TILES = 16
NUM_ACTIONS = 4

WEIGHT_INIT_SCALE = 0.01
INIT_LEARNING_RATE = 0.001
LR_DECAY_PER_100K = 0.5


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class FeedModel(object):
  """Class to construct and collect all relevant tensors of the model."""

  def __init__(self):
    self.state_batch_placeholder = tf.placeholder(
        tf.float32, shape=(None, NUM_TILES))
    self.targets_placeholder = tf.placeholder(tf.float32, shape=(None,))
    self.actions_placeholder = tf.placeholder(tf.int32, shape=(None,))
    self.placeholders = (self.state_batch_placeholder,
                         self.targets_placeholder,
                         self.actions_placeholder)

    self.q_values = build_inference_graph(self.state_batch_placeholder,
                                          [30, 20, 20])
    self.loss = build_loss(self.q_values, self.targets_placeholder,
                     self.actions_placeholder)
    self.train_op, self.global_step = build_train_op(self.loss)

    self.init = tf.initialize_all_variables()
    self.summary_op = tf.merge_all_summaries()



def build_inference_graph(state_batch, hidden_sizes):
  """Build inference model.

  Args:
    state_batch: [batch_size, NUM_TILES] float Tensor.
    hidden_sizes: Array of numbers where len(hidden_sizes) is the number of
        hidden layers and hidden_sizes[i] is the number of hidden units in the
        ith layer.

  Returns:
    q_values: Output tensor with the computed Q-Values.
  """
  input_batch = state_batch
  input_size = NUM_TILES

  for i, hidden_size in enumerate(hidden_sizes):
    hidden_output_i = build_fully_connected_layer(
        'hidden' + str(i), input_batch, input_size, hidden_size)

    input_batch = hidden_output_i
    input_size = hidden_size

  return build_fully_connected_layer('q_values', input_batch, input_size,
                                     NUM_ACTIONS)


def build_fully_connected_layer(name, input_batch, input_size, layer_size):
  """Builds a fully connected ReLU layer.

  Args:
    name: Name of the layer (-> Variable scope).
    input_batch: [batch_size, input_size] Tensor that this layer is
        connected to.
    input_size: Number of input units.
    layer_size: Number of units in this layer.

  Returns:
    The [batch_size, layer_size] output_batch Tensor.
  """
  with tf.name_scope(name):
    weights = tf.Variable(tf.truncated_normal([input_size, layer_size],
                                              stddev=WEIGHT_INIT_SCALE),
                          name='weights')
    biases = tf.Variable(tf.zeros([layer_size]), name='biases')
    output_batch = tf.nn.relu(tf.matmul(input_batch, weights) + biases)
    return output_batch


def build_loss(q_values, targets, actions):
  """Calculates the loss from the Q-Values, targets and actions.

  Args:
    q_values: A [batch_size, NUM_ACTIONS] float Tensor. Contains the current
        computed Q-Values.
    targets: A [batch_size] float Tensor. Contains the current target Q-Values
        for the action taken.
    actions: A [batch_size] int Tensor. Contains the actions taken.

  Returns:
    loss: Loss tensor of type float.
  """
  tf.scalar_summary("Average Target", tf.reduce_mean(targets))

  # Get Q-Value prodections for the given actions
  batch_size = tf.shape(q_values)[0]
  q_value_indices = tf.range(0, batch_size) * NUM_ACTIONS + actions
  relevant_q_values = tf.gather(tf.reshape(q_values, [-1]), q_value_indices)

  # Compute L2 loss (tf.nn.l2_loss() doesn't seem to be available on CPU)
  return tf.reduce_sum(tf.pow(relevant_q_values - targets, 2)) / 2


def build_train_op(loss):
  """Sets up the training Ops.

  Args:
    loss: Loss tensor, from loss().

  Returns:
    train_op, global_step: The Op for training & global step Tensor.
  """
  tf.scalar_summary("Loss", loss)

  global_step = tf.Variable(0, name='global_step', trainable=False)
  learning_rate = tf.train.exponential_decay(
      INIT_LEARNING_RATE, global_step, 100000, LR_DECAY_PER_100K)
  tf.scalar_summary("Learning Rate", learning_rate)

  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op, global_step
