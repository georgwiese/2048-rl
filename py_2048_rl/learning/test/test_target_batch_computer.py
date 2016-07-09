"""Tests for TargetBatchComputer."""

from py_2048_rl.learning import target_batch_computer

from mock import Mock, patch

import numpy as np

# pylint: disable=missing-docstring


def test_compute():
  target_batch_computer.GAMMA = 0.5
  target_batch_computer.MERGED_REWARD_FACTOR = 0.1
  target_batch_computer.LOST_REWARD = -1.0

  run_inference = Mock(return_value=np.array([[0, -0.2, -0.4, -0.6],
                                              [0, -0.1, -0.3, -0.5]]))
  computer = target_batch_computer.TargetBatchComputer(run_inference)

  reward_batch = np.array([1, 2])
  bad_action_batch = np.array([False, True])
  next_state_batch = np.arange(32).reshape((2, 16))
  available_actions_batch = np.array([[False, True, True, False],
                                      [True, False, False, True]])
  merged_batch = np.array([2, 0])

  target_batch = computer.compute(reward_batch, bad_action_batch,
                                  next_state_batch, available_actions_batch,
                                  merged_batch)

  assert (run_inference.call_args_list[0][0][0] == next_state_batch).all()

  # Targets should be:
  #   2 * 0.1 + 0.5 * (-0.2) = 0.1, and
  #   -1
  assert (target_batch == np.array([0.1, -1])).all()
