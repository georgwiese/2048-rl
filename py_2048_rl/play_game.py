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


def average_score(strategy):
  """Plays 100 games, returns average score."""

  scores = []
  for _ in range(100):
    score, _ = play.play(strategy)
    scores.append(score)
  return np.mean(scores)


def make_greedy_strategy(train_dir):
  """Load the latest checkpoint from train_dir, make a greedy strategy."""

  session = tf.Session()
  model = FeedModel()
  saver = tf.train.Saver()
  saver.restore(session, tf.train.latest_checkpoint(train_dir))

  get_q_values = learning.make_get_q_values(session, model)
  greedy_strategy = play.make_greedy_strategy(get_q_values)

  return greedy_strategy


def play_game():
  """Play a single game using the latest model snapshot in TRAIN_DIR."""

  print("Average Score: ", average_score(make_greedy_strategy(TRAIN_DIR)))


def main(_):
  """Main function."""

  play_game()


if __name__ == '__main__':
  tf.app.run()
