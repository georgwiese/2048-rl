"""Script to play a single game from a checkpoint."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from py_2048_rl.game import play
from py_2048_rl.learning import learning
from py_2048_rl.learning.model import FeedModel

import tensorflow as tf
import numpy as np

TRAIN_DIR = "/Users/georg/coding/2048-rl/train"

def play_game():

  with tf.Session() as session:
    model = FeedModel()
    saver = tf.train.Saver()
    saver.restore(session, tf.train.latest_checkpoint(TRAIN_DIR))

    get_q_values = learning.make_get_q_values(session, model)
    greedy_strategy = play.make_greedy_strategy(get_q_values)

    play.play(greedy_strategy, verbose=True)


def main(_):
  """Main function."""

  play_game()


if __name__ == '__main__':
  tf.app.run()
