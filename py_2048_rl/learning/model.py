"""Model"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow.python.platform
import tensorflow as tf


NUM_TILES = 16
NUM_ACTIONS = 4


def inference(state_batch, hidden1_units, hidden2_units):
  """Build inference model.

  Args:
    state_batch: [batch_size, NUM_TILES] float Tensor.
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.

  Returns:
    q_values: Output tensor with the computed Q-Values.
  """
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([NUM_TILES, hidden1_units], stddev=0.01),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]), name='biases')
    hidden1 = tf.nn.relu(tf.matmul(state_batch, weights) + biases)

  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units], stddev=0.01),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]), name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  with tf.name_scope('q_values'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_ACTIONS], stddev=0.01),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_ACTIONS]), name='biases')
    q_values = tf.matmul(hidden2, weights) + biases
  return q_values


def loss(q_values, targets, actions):
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
  # Get Q-Value prodections for the given actions
  batch_size = tf.shape(q_values)[0]
  q_value_indices = tf.range(0, batch_size) * NUM_ACTIONS + actions
  relevant_q_values = tf.gather(tf.reshape(q_values, [-1]), q_value_indices)

  # Compute L2 loss (tf.nn.l2_loss() doesn't seem to be available on CPU)
  return tf.reduce_sum(tf.pow(relevant_q_values - targets, 2)) / 2


def training(loss, learning_rate):
  """Sets up the training Ops.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  tf.scalar_summary("Loss", loss)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  global_step = tf.Variable(0, name='global_step', trainable=False)
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op
