"""Algorithms and strategies to play 2048 and collect experience."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from py_2048_rl.game.game import Game, ACTION_NAMES

import numpy as np

# pylint: disable=too-many-arguments,too-few-public-methods
class Experience(object):
  """Struct to encapsulate the experience of a single turn."""

  def __init__(self, state, action, reward, next_state, game_over,
               not_available, next_state_available_actions):
    """Initialize Experience

    Args:
      state: Shape (4, 4) numpy array, the state before the action was executed
      action: Number in range(4), action that was taken
      reward: Number, experienced reward
      next_state: Shape (4, 4) numpy array, the state after the action was
          executed
      game_over: boolean, whether next_state is a terminal state
      not_available: boolean, whether action was not available from state
      next_state_available_actions: Available actions from the next state
    """
    self.state = state
    self.action = action
    self.reward = reward
    self.next_state = next_state
    self.game_over = game_over
    self.not_available = not_available
    self.next_state_available_actions = next_state_available_actions

  def __str__(self):
    return str((self.state, self.action, self.reward, self.next_state,
                self.game_over, self.next_state_available_actions))

  def __repr__(self):
    return self.__str__()


def play(strategy, verbose=False, allow_unavailable_action=True):
  """Plays a single game, using a provided strategy.

  Args:
    strategy: A function that takes as argument a state and a list of available
        actions and returns an action from the list.
    allow_unavailable_action: Boolean, whether strategy is passed all actions
        or just the available ones.
    verbose: If true, prints game states, actions and scores.

  Returns:
    score, experiences where score is the final score and experiences is the
        list Experience instances that represent the collected experience.
  """

  game = Game()

  state = game.state().copy()
  game_over = game.game_over()
  experiences = []

  while not game_over:
    if verbose:
      print("Score:", game.score())
      game.print_state()

    old_state = state
    next_action = strategy(
        old_state, range(4) if allow_unavailable_action
                            else game.available_actions())

    if game.is_action_available(next_action):

      reward = game.do_action(next_action)
      state = game.state().copy()
      game_over = game.game_over()

      if verbose:
        print("Action:", ACTION_NAMES[next_action])
        print("Reward:", reward)

      experiences.append(Experience(old_state, next_action, reward, state,
                                    game_over, False, game.available_actions()))

    else:
      experiences.append(Experience(state, next_action, 0, state, False, True,
                                    game.available_actions()))

  if verbose:
    print("Score:", game.score())
    game.print_state()
    print("Game over.")

  return game.score(), experiences


def random_strategy(_, actions):
  """Strategy that always chooses actions at random."""
  return np.random.choice(actions)


def static_preference_strategy(_, actions):
  """Always prefer left over up over right over top."""

  return min(actions)


def highest_reward_strategy(state, actions):
  """Strategy that always chooses the action of highest immediate reward.

  If there are any ties, the strategy prefers left over up over right over down.
  """

  sorted_actions = np.sort(actions)[::-1]
  rewards = map(lambda action: Game(np.copy(state)).do_action(action),
                sorted_actions)
  action_index = np.argsort(rewards, kind="mergesort")[-1]
  return sorted_actions[action_index]

def make_greedy_strategy(get_q_values, verbose=False):
  """Makes greedy_strategy."""

  def greedy_strategy(state, actions):
    """Strategy that always picks the action of maximum Q(state, action)."""
    q_values = get_q_values(state)
    if verbose:
      print("State:")
      print(state)
      print("Q-Values:")
      for action, q_value, action_name in zip(range(4), q_values, ACTION_NAMES):
        not_available_string = "" if action in actions else "(not available)"
        print("%s:\t%.2f %s" % (action_name, q_value, not_available_string))
    sorted_actions = np.argsort(q_values)
    action = [a for a in sorted_actions if a in actions][-1]
    if verbose:
      print("-->", ACTION_NAMES[action])
    return action

  return greedy_strategy


def make_epsilon_greedy_strategy(get_q_values, epsilon):
  """Makes epsilon_greedy_strategy."""

  greedy_strategy = make_greedy_strategy(get_q_values)

  def epsilon_greedy_strategy(state, actions):
    """Picks random action with prob. epsilon, otherwise greedy_strategy."""
    do_random_action = np.random.choice([True, False], p=[epsilon, 1 - epsilon])
    if do_random_action:
      return random_strategy(state, actions)
    return greedy_strategy(state, actions)

  return epsilon_greedy_strategy
