"""Learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math
import random
from collections import deque

from py_2048_rl.game import play
from py_2048_rl.learning.model import FeedModel

import tensorflow as tf
import numpy as np

BATCH_SIZE = 32

EXPERIENCE_SIZE = 10000
STATE_NORMALIZE_FACTOR = 1.0 / 15.0
REWARD_NORMALIZE_FACTOR = 1.0 / 25.0

GAMMA = 0.0

MEMORY_CAPACITY = 1e4
START_DECREASE_EPSILON_GAMES = 200000
DECREASE_EPSILON_GAMES = 100000.0
MIN_EPSILON = 1.0
BATCHES_KEEP_CONSTANT = 1e3
NOT_LOST_KEEP_PROB = 1.0

RESUME = False
TRAIN_DIR = "./train"

def collect_experience(strategy, num_games=1):
  """Plays num_games random games, returns all collected experiences."""

  experiences = []
  for _ in range(num_games):
    _, new_experiences = play.play(strategy)
    experiences += [e for e in new_experiences
                    if e.game_over or np.random.rand() < NOT_LOST_KEEP_PROB]
  return experiences


def add_to_memory(memory, experience):
  """Add a single experience to memory."""

  memory.append(experience)
  if len(memory) > MEMORY_CAPACITY:
    memory.popleft()


def get_batches_stepwise(get_q_values, run_inference):
  """Wraps get_batches(), keeping the current predictions constant for
  BATCHES_KEEP_CONSTANT steps.
  """

  cache = []

  for batches in get_batches(get_q_values, run_inference):
    cache.append(batches)

    if len(cache) >= BATCHES_KEEP_CONSTANT:
      for cached_batches in cache:
        yield cached_batches
      cache = []


def get_batches(get_q_values, run_inference):
  """Yields randomized batches epsilon-greedy games.

  Maintains a replay memory at full capacity.
  """

  print("Initializing memory...")
  memory = deque()
  while len(memory) < MEMORY_CAPACITY:
    for experience in collect_experience(play.random_strategy):
      add_to_memory(memory, experience)

  for i in itertools.count():
    if i < START_DECREASE_EPSILON_GAMES:
      epsilon = 1.0
    else:
      epsilon = max(MIN_EPSILON,
                    1.0 - (i - START_DECREASE_EPSILON_GAMES) /
                    DECREASE_EPSILON_GAMES)

    strategy = play.make_epsilon_greedy_strategy(get_q_values, epsilon)

    for experience in collect_experience(strategy):
      add_to_memory(memory, experience)
      batch_experiences = random.sample(memory, BATCH_SIZE)
      yield experiences_to_batches(batch_experiences, run_inference)


def experiences_to_batches(experiences, run_inference):
  """Computes state_batch, targets, actions."""

  batch_size = len(experiences)
  state_batch = np.zeros((batch_size, 16))
  next_state_batch = np.zeros((batch_size, 16))
  targets = np.zeros((batch_size,), dtype=np.float)
  actions = np.zeros((batch_size,), dtype=np.int)
  bad_action_batch = np.zeros((batch_size,), dtype=np.bool)

  for i, experience in enumerate(experiences):
    state_batch[i, :] = experience.state.flatten() * STATE_NORMALIZE_FACTOR
    next_state_batch[i, :] = (experience.next_state.flatten() *
                              STATE_NORMALIZE_FACTOR)
    actions[i] = experience.action
    bad_action_batch[i] = experience.game_over or experience.not_available
    targets[i] = np.count_nonzero(experience.state) - np.count_nonzero(experience.next_state)

  good_action_batch = np.logical_not(bad_action_batch)

  # targets[bad_action_batch] = -1
  # targets[good_action_batch] = 0

  if GAMMA > 0:
    predictions = run_inference(next_state_batch)
    max_qs = predictions.max(axis=1)
    targets[good_action_batch] += GAMMA * max_qs[good_action_batch]

  return state_batch, targets, actions


def make_run_inference(session, model):
  """Make run_inference() function for given session and model."""

  def run_inference(state_batch):
    """Run inference on a given state_batch. Returns a q value batch."""
    return session.run(model.q_values,
                       feed_dict={model.state_batch_placeholder: state_batch})
  return run_inference


def make_get_q_values(session, model):
  """Make get_q_values() function for given session and model."""

  run_inference = make_run_inference(session, model)
  def get_q_values(state):
    """Run inference on a single (4, 4) state matrix."""
    state_vector = state.flatten() * STATE_NORMALIZE_FACTOR
    state_batch = np.array([state_vector])
    q_values_batch = run_inference(state_batch)
    return q_values_batch[0]
  return get_q_values


def run_training():
  """Run training"""

  print("Train dir: ", TRAIN_DIR)

  with tf.Graph().as_default():
    model = FeedModel()
    saver = tf.train.Saver()
    session = tf.Session()
    summary_writer = tf.train.SummaryWriter(TRAIN_DIR,
                                            graph_def=session.graph_def,
                                            flush_secs=10)

    if RESUME:
      saver.restore(session, tf.train.latest_checkpoint(TRAIN_DIR))
    else:
      session.run(model.init)

    run_inference = make_run_inference(session, model)
    get_q_values = make_get_q_values(session, model)

    test_experiences = collect_experience(play.random_strategy, 100)

    for state_batch, targets, actions in get_batches_stepwise(
        get_q_values, run_inference):

      global_step, _ = session.run([model.global_step, model.train_op],
          feed_dict={
              model.state_batch_placeholder: state_batch,
              model.targets_placeholder: targets,
              model.actions_placeholder: actions,})

      if global_step % 10000 == 0 and global_step != 0:
        saver.save(session, TRAIN_DIR + "/checkpoint", global_step=global_step)
        loss = write_summaries(session, run_inference, model, test_experiences,
                               summary_writer)
        print("Step:", global_step, "Loss:", loss)


def write_summaries(session, run_inference, model, test_experiences,
                    summary_writer):
  """Writes summaries by running the model on test_experiences. Returns loss."""

  state_batch, targets, actions = experiences_to_batches(
      test_experiences, run_inference)
  state_batch_p, targets_p, actions_p = model.placeholders
  summary_str, global_step, loss = session.run(
      [model.summary_op, model.global_step, model.loss],
      feed_dict={
          state_batch_p: state_batch,
          targets_p: targets,
          actions_p: actions,})
  summary_writer.add_summary(summary_str, global_step)
  return loss


def main(_):
  """Main function."""

  run_training()


if __name__ == '__main__':
  tf.app.run()
