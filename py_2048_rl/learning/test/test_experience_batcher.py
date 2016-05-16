"""Tests for ExperienceBatcher."""

from py_2048_rl.game.play import Experience
from py_2048_rl.learning.experience_batcher import ExperienceBatcher

from mock import Mock, patch

import numpy as np

# pylint: disable=missing-docstring


@patch('py_2048_rl.learning.experience_batcher.TargetBatchComputer')
def test_experiences_to_batches(target_computer_class_mock):
  compute = target_computer_class_mock.return_value.compute
  compute.return_value = np.array([42, 43])

  state1 = np.arange(16).reshape((4, 4)) + 1
  state2 = np.arange(16).reshape((4, 4)) + 2
  state3 = np.arange(16).reshape((4, 4)) + 3
  experiences = [Experience(state1, 1, 2, state2, False, False, [3]),
                 Experience(state2, 3, 4, state3, True, False, [])]

  run_inference = Mock(side_effect=[np.array([[0, 0, 0, -0.5],
                                              [0, 0, 0, 0]])])

  batcher = ExperienceBatcher(None, run_inference, None, 1.0 / 15.0)

  state_batch, targets, actions = batcher.experiences_to_batches(experiences)

  reward_batch = np.array([2, 4])
  bad_action_batch = np.array([False, True])
  next_state_batch = np.array([state2.flatten(), state3.flatten()]) / 15.0
  available_actions_batch = np.array([[False, False, False, True],
                                      [False, False, False, False]])

  assert (compute.call_args_list[0][0][0] == reward_batch).all()
  assert (compute.call_args_list[0][0][1] == bad_action_batch).all()
  assert (compute.call_args_list[0][0][2] == next_state_batch).all()
  assert (compute.call_args_list[0][0][3] == available_actions_batch).all()

  expected_state_batch = np.array([state1.flatten(), state2.flatten()]) / 15.0

  assert (state_batch == expected_state_batch).all()
  assert (targets == np.array([42, 43])).all()
  assert (actions == np.array([1, 3])).all()
