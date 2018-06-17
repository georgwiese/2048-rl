"""Provides functions to compute the target value given experience batches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


GAMMA = 0.5
MERGED_REWARD_FACTOR = 1.0
LOST_REWARD = 0.0


class TargetBatchComputer(object):
  """Computes the target batch for the neural network."""

  def __init__(self, run_inference):
    """Init TargetBatchComputer.

    Args:
      run_inference: function (state batch) -> estimated Q-Values
    """
    self.run_inference = run_inference


  def compute(self, reward_batch, bad_action_batch, next_state_batch,
              available_actions_batch, merged):
    """Computes the target batch for the neural network.

    Args:
      reward_batch: A (batch_size,) float numpy array containing the rewards from
          the game associated with the experiences in the batch.
      bad_action_batch: A (batch_size,) bool numpy array containing whether the
          respective experience was "bad" (i.e. lost the game).
      next_state_batch: A (batch_size, 16) float numpy array where each row
          contains the next state associated with the experience. The values MUST
          be already in the right scale so that they can be passed directly to
          the network for inference.
      available_actions_batch: A (batch_size, 4) bool array that stores which
          actions are available from the next state.
      merged: A (batch_size,) float numpy array that contains how many tiles
          have been merged at tha current experience.

    Returns:
      A (batch_size,) float numpy array that contains the target values for the
          current batch.
    """

    (batch_size,) = reward_batch.shape
    targets = np.zeros((batch_size,), dtype=np.float)

    good_action_batch = np.logical_not(bad_action_batch)

    targets[bad_action_batch] = LOST_REWARD
    targets[good_action_batch] = (merged[good_action_batch] *
                                  MERGED_REWARD_FACTOR)

    if GAMMA > 0:
      predictions = self.run_inference(next_state_batch)
      # Remove non-available predictions
      predictions[np.logical_not(available_actions_batch)] = -1e8
      max_qs = predictions.max(axis=1)
      max_qs = np.maximum(max_qs, -1)
      targets[good_action_batch] += GAMMA * max_qs[good_action_batch]

    return targets
