"""Script to analyze a given model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from py_2048_rl.game import play
from py_2048_rl.learning import learning
from py_2048_rl.learning.experience_collector import ExperienceCollector
from py_2048_rl.learning.model import FeedModel

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

TRAIN_DIR = "./train"


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


def analyze():
  """Plot all Q-Values."""

  plt.hist(get_all_q_values(TRAIN_DIR))
  plt.show()


def main(_):
  """Main function."""

  analyze()


if __name__ == '__main__':
  tf.app.run()
