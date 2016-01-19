"""Learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from py_2048_rl.game.play import play, random_strategy, make_greedy_strategy, make_epsilon_greedy_strategy
from py_2048_rl.learning import model

import tensorflow.python.platform
import tensorflow as tf

import numpy as np

BATCH_SIZE = 20
LEARNING_RATE = 0.0001

EXPERIENCE_SIZE = 100000
STATE_NORMALIZE_FACTOR = 1.0 / 15.0
REWARD_NORMALIZE_FACTOR = 1.0 / 1000.0

TRAIN_DIR = "/Users/georg/coding/2048-rl/train2"

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
    epsilon = max(0, 1.0 - i / 100000.0)
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
    state_batch_placeholder = tf.placeholder(
        tf.float32, shape=(None, model.NUM_TILES))
    targets_placeholder = tf.placeholder(tf.float32, shape=(None,))
    actions_placeholder = tf.placeholder(tf.int32, shape=(None,))
    placeholders = (state_batch_placeholder, targets_placeholder,
                    actions_placeholder)

    q_values = model.inference(state_batch_placeholder, 20, 20)
    loss = model.loss(q_values, targets_placeholder, actions_placeholder)
    train_op = model.training(loss, LEARNING_RATE)

    summary_op = tf.merge_all_summaries()
    saver = tf.train.Saver()
    sess = tf.Session()

    init = tf.initialize_all_variables()
    sess.run(init)

    summary_writer = tf.train.SummaryWriter(TRAIN_DIR,
                                            graph_def=sess.graph_def,
                                            flush_secs=10)

    def run_inference(state_batch):
      """Run inference"""
      return sess.run(q_values,
                      feed_dict={state_batch_placeholder: state_batch})

    def get_q_values(state):
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

        feed_dict = {
            state_batch_placeholder: state_batch,
            targets_placeholder: targets,
            actions_placeholder: actions,
        }
        [loss_value, _] = sess.run([loss, train_op], feed_dict=feed_dict)
        loss_sum += loss_value

        duration = time.time() - start_time

        if global_step % 500 == 0 and global_step != 0:
          avg_loss = loss_sum / 500
          loss_sum = 0
          print('Step %d: Games: %d loss = %.6f (%.3f sec), avg target: %f' % (
              global_step, (n_round+1) * 100, avg_loss, duration,
              np.average(targets)))

        if global_step % 1000 == 0 and global_step != 0:
          saver.save(sess, TRAIN_DIR + "/checkpoint", global_step=global_step)
          print('Average Score: %f' % evaluate(get_q_values, sess, placeholders,
                                               test_experiences, summary_op,
                                               run_inference,
                                               summary_writer, global_step,
                                               global_step % 10000 == 0))

        global_step += 1


def evaluate(get_q_values, session, placeholders, test_experiences, summary_op,
             run_inference, summary_writer, global_step, verbose=False):

  state_batch, targets, actions = get_batches(test_experiences, run_inference)
  state_batch_p, targets_p, actions_p = placeholders
  summary_str = session.run(summary_op, feed_dict={
      state_batch_p: state_batch,
      targets_p: targets,
      actions_p: actions,
  })
  summary_writer.add_summary(summary_str, global_step)

  greedy_strategy = make_greedy_strategy(get_q_values)

  if verbose:
    play(greedy_strategy, True)

  scores = []
  for _ in range(100):
    score, _ = play(greedy_strategy)
    scores.append(score)
  return np.average(scores)


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
