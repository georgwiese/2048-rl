"""Script to analyze a given model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from py_2048_rl.game import play
from py_2048_rl.learning import learning
from py_2048_rl.learning.experience_collector import ExperienceCollector
from py_2048_rl.learning.model import FeedModel


def get_all_q_values(train_dir):
  """Play randomly, compute q-values for all states."""

  session = tf.Session()
  model = FeedModel()
  saver = tf.train.Saver()
  saver.restore(session, tf.train.latest_checkpoint(train_dir))

  get_q_values = learning.make_get_q_values(session, model)
  experiences = ExperienceCollector().collect(play.random_strategy, 100)

  all_q_values = []
  for experience in experiences:
    all_q_values += list(get_q_values(experience.next_state))

  return all_q_values


def analyze(train_dir):
  """Plot all Q-Values."""

  plt.hist(get_all_q_values(train_dir))
  plt.show()


def main(args):
  """Main function."""

  if len(args) != 2:
    print("Usage: %s train_dir" % args[0])
    sys.exit(1)

  analyze(args[1])


if __name__ == '__main__':
  tf.app.run()
