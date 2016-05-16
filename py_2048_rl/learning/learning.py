"""Learning algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os

import tensorflow as tf
import numpy as np

from py_2048_rl.game import play
from py_2048_rl.learning.experience_batcher import ExperienceBatcher
from py_2048_rl.learning.experience_collector import ExperienceCollector
from py_2048_rl.learning.model import FeedModel


STATE_NORMALIZE_FACTOR = 1.0 / 15.0


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


def run_training(train_dir):
  """Run training"""

  resume = os.path.exists(train_dir)

  with tf.Graph().as_default():
    model = FeedModel()
    saver = tf.train.Saver()
    session = tf.Session()
    summary_writer = tf.train.SummaryWriter(train_dir,
                                            graph_def=session.graph_def,
                                            flush_secs=10)

    if resume:
      print("Resuming: ", train_dir)
      saver.restore(session, tf.train.latest_checkpoint(train_dir))
    else:
      print("Starting new training: ", train_dir)
      session.run(model.init)

    run_inference = make_run_inference(session, model)
    get_q_values = make_get_q_values(session, model)

    experience_collector = ExperienceCollector()
    batcher = ExperienceBatcher(experience_collector, run_inference,
                                get_q_values, STATE_NORMALIZE_FACTOR)

    test_experiences = experience_collector.collect(play.random_strategy, 100)

    for state_batch, targets, actions in batcher.get_batches_stepwise():

      global_step, _ = session.run([model.global_step, model.train_op],
          feed_dict={
              model.state_batch_placeholder: state_batch,
              model.targets_placeholder: targets,
              model.actions_placeholder: actions,})

      if global_step % 10000 == 0 and global_step != 0:
        saver.save(session, train_dir + "/checkpoint", global_step=global_step)
        loss = write_summaries(session, batcher, model, test_experiences,
                               summary_writer)
        print("Step:", global_step, "Loss:", loss)


def write_summaries(session, batcher, model, test_experiences, summary_writer):
  """Writes summaries by running the model on test_experiences. Returns loss."""

  state_batch, targets, actions = batcher.experiences_to_batches(
      test_experiences)
  state_batch_p, targets_p, actions_p = model.placeholders
  summary_str, global_step, loss = session.run(
      [model.summary_op, model.global_step, model.loss],
      feed_dict={
          state_batch_p: state_batch,
          targets_p: targets,
          actions_p: actions,})
  summary_writer.add_summary(summary_str, global_step)
  return loss


def main(args):
  """Main function."""

  if len(args) != 2:
    print("Usage: %s train_dir" % args[0])
    sys.exit(1)

  run_training(args[1])


if __name__ == '__main__':
  tf.app.run()
