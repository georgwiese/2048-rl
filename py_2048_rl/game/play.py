"""Algorithms to play 2048 and collect experience."""

from py_2048_rl.game.game import Game, ACTION_NAMES

import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
class Experience(object):
  """Struct to encapsulate the experience of a single turn."""

  def __init__(self, state, action, reward, next_state, game_over):
    """Initialize Experience

    Args:
      state: Shape (4, 4) numpy array, the state before the action was executed
      action: Number in range(4), action that was taken
      reward: Number, experienced reward
      next_state: Shape (4, 4) numpy array, the state after the action was
          executed
      game_over: boolean, whether next_state is a terminal state
    """
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.game_over = game_over

  def __str__(self):
    return str((self.state, self.action, self.reward, self.next_state,
                self.game_over))

  def __repr__(self):
    return self.__str__()


def strategy_random(_, actions):
  """Strategy that always chooses actions at random."""
  return np.random.choice(actions)


def play(strategy, verbose=False):
  """Plays a single game, using a provided strategy.

  Args:
    strategy: A function that takes as argument a state and a list of available
        actions and returns an action from the list.
    verbose: If true, prints game states, actions and scores.
  """

  game = Game()

  state = game.state().copy()
  game_over = game.game_over()
  experiences = []

  while not game_over:
    old_state = state
    actions = game.available_actions()
    next_action = strategy(old_state, actions)

    if verbose:
      print "Score:", game.score()
      game.print_state()
      print "Action:", ACTION_NAMES[next_action]

    reward = game.do_action(next_action)
    state = game.state().copy()
    game_over = game.game_over()

    experiences.append(Experience(old_state, next_action, reward, state,
                                  game_over))

  if verbose:
    print "Score:", game.score()
    game.print_state()
    print "Game over."

  return game.score(), experiences
