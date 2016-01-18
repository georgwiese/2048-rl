"""Tests for `Game` class."""

from py_2048_rl.game.game import Game
from mock import call, patch

import numpy as np

# pylint: disable=missing-docstring

@patch('numpy.random.choice')
def test_init(choice):
  choice.side_effect = [0,  # First position
                        1,  # First tile
                        1,  # Second position
                        2]  # Second tile
  game = Game()

  choice.assert_has_calls([call(16),
                           call([1, 2], p=[0.9, 0.1]),
                           call(15),
                           call([1, 2], p=[0.9, 0.1])])

  # Assert correct number of 0s, 1s and 2s
  game.print_state()
  assert (np.bincount(game.state().flatten()) == [14, 1, 1]).all()
  assert game.score() == 0


def test_available_actions():
  state = np.array([[1, 2, 3, 0],
                    [1, 2, 3, 0],
                    [1, 2, 3, 0],
                    [1, 2, 3, 0]])

  game = Game(state=state)
  actions = game.available_actions()

  # All actions except left is available
  assert actions == [1, 2, 3]


def test_available_actions_none_available():
  state = np.array([[1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [1, 2, 3, 4],
                    [5, 6, 7, 8]])

  game = Game(state=state)
  actions = game.available_actions()

  # All actions except left is available
  assert actions == []
  assert game.game_over()


@patch('numpy.random.choice')
def test_do_action(choice):
  choice.side_effect = [0,  # First position
                        1]  # First tile
  state = np.array([[1, 2, 3, 3],
                    [5, 6, 7, 8],
                    [5, 2, 7, 0],
                    [1, 0, 3, 0]])

  game = Game(state=state)
  game.do_action(3)  # DOWN

  new_state = np.array([[1, 0, 0, 0],
                        [1, 2, 3, 0],
                        [6, 6, 8, 3],
                        [1, 2, 3, 8]])
  game.print_state()
  assert (game.state() == new_state).all()
  # Score is 2 ** 6 + 2 ** 8
  assert game.score() == 320
