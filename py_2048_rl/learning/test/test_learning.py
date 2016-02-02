"""Tests for Learning Algorithms."""

from py_2048_rl.game.play import Experience
from py_2048_rl.learning import learning

from mock import Mock

import numpy as np

# pylint: disable=missing-docstring

def test_experiences_to_batches():
  learning.BATCH_SIZE = 2
  learning.GAMMA = 0.5

  state1 = np.arange(16).reshape((4, 4)) + 1
  state2 = np.arange(16).reshape((4, 4)) + 2
  state3 = np.arange(16).reshape((4, 4)) + 3
  experiences = [Experience(state1, 1, 2, state2, False),
                 Experience(state2, 3, 4, state3, True)]

  run_inference = Mock(side_effect=[np.array([[1, 2, 3],
                                              [6, 5, 4]])])

  state_batch, targets, actions = learning.experiences_to_batches(
      experiences, run_inference)

  next_state_batch = np.array([state2.flatten(), state3.flatten()]) / 15.0
  assert (run_inference.call_args_list[0][0][0] == next_state_batch).all()

  expected_state_batch = np.array([state1.flatten(), state2.flatten()]) / 15.0
  assert (state_batch == expected_state_batch).all()

  # Targets should be 2.5 (1 reward + 0.5 * 3 max Q) and 1
  # (1 reward, final state).
  assert (targets == np.array([2.5, 1])).all()

  assert (actions == np.array([1, 3])).all()
