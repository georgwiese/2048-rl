"""Tests for the play module."""

from py_2048_rl.game.play import play
from mock import Mock, patch, call

import numpy as np

# pylint: disable=missing-docstring

@patch('py_2048_rl.game.play.Game')
def test_play(game_class_mock):
  state1 = np.ones((4, 4))
  state2 = np.ones((4, 4)) * 2
  state3 = np.ones((4, 4)) * 3

  game = game_class_mock.return_value
  game.game_over.side_effect = [False, False, True]
  game.state.side_effect = [state1, state2, state3]
  game.available_actions.side_effect = [[1, 2, 3], [0, 1, 2]]
  game.do_action.side_effect = [1, 2]
  game.score.return_value = 1234

  strategy = Mock(side_effect=[1, 2])

  score, experiences = play(strategy)

  game.do_action.assert_has_calls([call(1), call(2)])
  # Manually need to check strategy arguments, because numpy array overrides
  # == operator...
  assert (strategy.call_args_list[0][0][0] == state1).all()
  assert strategy.call_args_list[0][0][1] == [1, 2, 3]
  assert (strategy.call_args_list[1][0][0] == state2).all()
  assert strategy.call_args_list[1][0][1] == [0, 1, 2]

  assert score == 1234

  assert len(experiences) == 2

  assert (experiences[0].state == state1).all()
  assert experiences[0].action == 1
  assert experiences[0].reward == 1
  assert (experiences[0].next_state == state2).all()
  assert experiences[0].game_over == False

  assert (experiences[1].state == state2).all()
  assert experiences[1].action == 2
  assert experiences[1].reward == 2
  assert (experiences[1].next_state == state3).all()
  assert experiences[1].game_over == True
