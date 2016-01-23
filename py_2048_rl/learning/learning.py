"""Learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from py_2048_rl.game import play
from py_2048_rl.learning.model import FeedModel

import tensorflow as tf

import numpy as np

BATCH_SIZE = 20

EXPERIENCE_SIZE = 100000
STATE_NORMALIZE_FACTOR = 1.0 / 15.0
REWARD_NORMALIZE_FACTOR = 1.0 / 1000.0

MIN_EPSILON = 0.1

TRAIN_DIR = "/Users/georg/coding/2048-rl/train"

def collect_experience(num_games, strategy):
  """Plays num_games random games, returns all collected experiences."""

  experiences = []
  for _ in range(num_games):
    _, new_experiences = play.play(strategy)
    experiences += new_experiences
  return experiences


def get_batches(get_q_values, run_inference):
  """Yields training batches from 100 epsilon-greedy games, in random order."""

  for i in itertools.count():
    epsilon = max(MIN_EPSILON, 1.0 - i / 1000.0)
    print("Collecting experience, epsilon: %f" % epsilon)
    print("Games: %d" % ((i + 1) * 100))

    strategy = play.make_epsilon_greedy_strategy(get_q_values, epsilon)
    experiences = collect_experience(100, strategy)

    steps = len(experiences) // BATCH_SIZE
    experience_indices = np.random.permutation(len(experiences))

    for step in range(steps):
      batch_indices = experience_indices[step * BATCH_SIZE :
                                         (step + 1) * BATCH_SIZE]
      batch_experiences = [experiences[i] for i in batch_indices]

      yield experiences_to_batches(batch_experiences, run_inference)


def experiences_to_batches(experiences, run_inference):
  """Computes state_batch, targets, actions."""

  batch_size = len(experiences)
  state_batch = np.zeros((batch_size, 16))
  next_state_batch = np.zeros((batch_size, 16))
  targets = np.zeros((batch_size,), dtype=np.float)
  actions = np.zeros((batch_size,), dtype=np.int)
  game_over_batch = np.zeros((batch_size,), dtype=np.bool)

  for i, experience in enumerate(experiences):
    state_batch[i, :] = experience.state.flatten() * STATE_NORMALIZE_FACTOR
    next_state_batch[i, :] = (experience.next_state.flatten() *
                              STATE_NORMALIZE_FACTOR)
    actions[i] = experience.action
    targets[i] = experience.reward * REWARD_NORMALIZE_FACTOR
    game_over_batch[i] = experience.game_over

  predictions = run_inference(next_state_batch)
  max_qs = predictions.max(axis=1)
  max_qs[game_over_batch] = 0
  targets += max_qs

  return state_batch, targets, actions


def run_training():
  """Run training"""

  with tf.Graph().as_default():
    model = FeedModel()
    saver = tf.train.Saver()
    session = tf.Session()
    summary_writer = tf.train.SummaryWriter(TRAIN_DIR,
                                            graph_def=session.graph_def,
                                            flush_secs=10)

    session.run(model.init)

    def run_inference(state_batch):
      """Run inference on a given state_batch. Returns a q value batch."""
      return session.run(model.q_values,
                         feed_dict={model.state_batch_placeholder: state_batch})

    def get_q_values(state):
      """Run inference an a single (4, 4) state matrix."""
      state_vector = state.flatten() * STATE_NORMALIZE_FACTOR
      state_batch = np.array([state_vector])
      q_values_batch = run_inference(state_batch)
      return q_values_batch[0]

    test_experiences = collect_experience(100, play.random_strategy)

    for state_batch, targets, actions in get_batches(
        get_q_values, run_inference):

      global_step, _ = session.run([model.global_step, model.train_op],
          feed_dict={
              model.state_batch_placeholder: state_batch,
              model.targets_placeholder: targets,
              model.actions_placeholder: actions,})

      if global_step % 10000 == 0 and global_step != 0:
        saver.save(session, TRAIN_DIR + "/checkpoint", global_step=global_step)
        write_summaries(session, run_inference, model, test_experiences,
                        summary_writer)
        print('Average Score: %f' % evaluate(get_q_values))


def write_summaries(session, run_inference, model, test_experiences,
                    summary_writer):
  """Writes summaries by running the model on test_experiences."""

  state_batch, targets, actions = experiences_to_batches(
      test_experiences, run_inference)
  state_batch_p, targets_p, actions_p = model.placeholders
  summary_str, global_step = session.run([model.summary_op, model.global_step],
      feed_dict={
          state_batch_p: state_batch,
          targets_p: targets,
          actions_p: actions,})
  summary_writer.add_summary(summary_str, global_step)


def evaluate(get_q_values, verbose=False):
  """Plays 100 games with greedy_strategy, returns average score."""

  greedy_strategy = play.make_greedy_strategy(get_q_values)

  if verbose:
    play.play(greedy_strategy, True)

  scores = []
  for _ in range(100):
    score, _ = play.play(greedy_strategy)
    scores.append(score)
  return np.average(scores)


def main(_):
  """Main function."""

  run_training()


if __name__ == '__main__':
  tf.app.run()
