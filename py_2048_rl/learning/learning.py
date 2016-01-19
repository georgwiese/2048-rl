"""Learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from py_2048_rl.game.play import play, strategy_random
from py_2048_rl.learning import model

import tensorflow.python.platform
import tensorflow as tf

import numpy as np

BATCH_SIZE = 20
LEARNING_RATE = 0.0001

EXPERIENCE_SIZE = 100000
STATE_NORMALIZE_FACTOR = 1.0 / 15.0
REWARD_NORMALIZE_FACTOR = 1.0 / 1000.0

TRAIN_DIR = "/Users/georg/coding/2048-rl/train"

def collect_experience(num_games):
  """Plays num_games random games, returns all collected experiences."""
  experiences = []
  for _ in range(num_games):
    _, new_experiences = play(strategy_random)
    experiences += new_experiences
  return experiences


def get_experiences():
  """Yields experiences from 100 random games."""
  while True:
    yield collect_experience(100)


def get_batches(experiences, run_inference):
  """Computes state_batch, targets, actions."""

  assert len(experiences) == BATCH_SIZE
  state_batch = np.zeros((BATCH_SIZE, 16))
  next_state_batch = np.zeros((BATCH_SIZE, 16))
  targets = np.zeros((BATCH_SIZE,), dtype=np.float)
  actions = np.zeros((BATCH_SIZE,), dtype=np.int)
  game_over_batch = np.zeros((BATCH_SIZE,), dtype=np.bool)

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
        tf.float32, shape=(BATCH_SIZE, model.NUM_TILES))
    targets_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE,))
    actions_placeholder = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))

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

    global_step = 0
    for n_round, experiences in enumerate(get_experiences()):

      steps = len(experiences) // BATCH_SIZE
      experience_indices = np.random.permutation(len(experiences))

      loss_sum = 0

      for step in range(steps):
        start_time = time.time()

        batch_indices = experience_indices[step * BATCH_SIZE :
                                           (step + 1) * BATCH_SIZE]
        batch_experiences = [experiences[i] for i in batch_indices]

        def run_inference(state_batch):
          """Run inference"""
          return sess.run(q_values,
                          feed_dict={state_batch_placeholder: state_batch})

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
          summary_str = sess.run(summary_op, feed_dict=feed_dict)
          summary_writer.add_summary(summary_str, global_step)

        if global_step % 1000 == 0 and global_step != 0:
          saver.save(sess, TRAIN_DIR + "/checkpoint", global_step=global_step)
          print('Average Score: %f' % evaluate(
              sess, q_values, state_batch_placeholder,
              global_step % 10000 == 0))

        global_step += 1


def evaluate(session, q_values_tensor, state_batch_placeholder, verbose=False):

  def greedy_strategy(state, actions):
    state_vector = state.flatten() * STATE_NORMALIZE_FACTOR
    state_batch = np.array([state_vector] * BATCH_SIZE)
    q_values = session.run(q_values_tensor,
                           feed_dict={state_batch_placeholder: state_batch})
    sorted_actions = np.argsort(q_values[0, :])
    return [a for a in sorted_actions if a in actions][0]

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
