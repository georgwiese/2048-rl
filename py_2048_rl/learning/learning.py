"""Learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from py_2048_rl.game.play import play, random_strategy, make_greedy_strategy, make_epsilon_greedy_strategy
from py_2048_rl.learning.model import FeedModel

import tensorflow as tf

import numpy as np

BATCH_SIZE = 20

EXPERIENCE_SIZE = 100000
STATE_NORMALIZE_FACTOR = 1.0 / 15.0
REWARD_NORMALIZE_FACTOR = 1.0 / 1000.0

TRAIN_DIR = "/Users/georg/coding/2048-rl/train"

def collect_experience(num_games, strategy):
  """Plays num_games random games, returns all collected experiences."""
  experiences = []
  for _ in range(num_games):
    _, new_experiences = play(strategy)
    experiences += new_experiences
  return experiences


def get_experiences(get_q_values):
  """Yields experiences from 100 random games."""
  i = 0
  while True:
    epsilon = max(0, 1.0 - i / 1000.0)
    print("Collecting experience, epsilon: %f" % epsilon)

    strategy = make_epsilon_greedy_strategy(get_q_values, epsilon)
    yield collect_experience(100, strategy)
    i += 1


def get_batches(experiences, run_inference):
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

    test_experiences = collect_experience(100, random_strategy)

    global_step = 0
    for n_round, experiences in enumerate(get_experiences(get_q_values)):

      steps = len(experiences) // BATCH_SIZE
      experience_indices = np.random.permutation(len(experiences))

      loss_sum = 0

      for step in range(steps):
        start_time = time.time()

        batch_indices = experience_indices[step * BATCH_SIZE :
                                           (step + 1) * BATCH_SIZE]
        batch_experiences = [experiences[i] for i in batch_indices]

        state_batch, targets, actions = get_batches(batch_experiences,
                                                    run_inference)

        [loss_value, _] = session.run(
            [model.loss, model.train_op],
            feed_dict={
                model.state_batch_placeholder: state_batch,
                model.targets_placeholder: targets,
                model.actions_placeholder: actions,})

        loss_sum += loss_value
        duration = time.time() - start_time

        if global_step % 500 == 0 and global_step != 0:
          avg_loss = loss_sum / 500
          loss_sum = 0
          print('Step %d: Games: %d loss = %.6f (%.3f sec), avg target: %f' % (
              global_step, (n_round+1) * 100, avg_loss, duration,
              np.average(targets)))

        if global_step % 1000 == 0 and global_step != 0:
          saver.save(session, TRAIN_DIR + "/checkpoint", global_step=global_step)
          write_summaries(session, run_inference, model, test_experiences,
                          summary_writer, global_step)
          print('Average Score: %f' % evaluate(get_q_values, verbose=True))

        global_step += 1


def write_summaries(session, run_inference, model, test_experiences,
                    summary_writer, global_step):
  """Writes summaries by running the model on test_experiences."""

  state_batch, targets, actions = get_batches(test_experiences, run_inference)
  state_batch_p, targets_p, actions_p = model.placeholders
  summary_str = session.run(model.summary_op, feed_dict={
      state_batch_p: state_batch,
      targets_p: targets,
      actions_p: actions,
  })
  summary_writer.add_summary(summary_str, global_step)


def evaluate(get_q_values, verbose=False):
  """Plays 100 games with greedy_strategy, returns average score."""

  greedy_strategy = make_greedy_strategy(get_q_values)

  if verbose:
    play(greedy_strategy, True)

  scores = []
  for _ in range(100):
    score, _ = play(greedy_strategy)
    scores.append(score)
  return np.average(scores)


def main(_):
  """Main function."""
  run_training()


if __name__ == '__main__':
  tf.app.run()
