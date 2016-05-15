"""Class that builds experience batches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import itertools

from py_2048_rl.game import play
from py_2048_rl.learning.replay_memory import ReplayMemory


BATCH_SIZE = 32
GAMMA = 0.00

START_DECREASE_EPSILON_GAMES = 200000
DECREASE_EPSILON_GAMES = 100000.0
MIN_EPSILON = 1.0
BATCHES_KEEP_CONSTANT = 1e3


class ExperienceBatcher(object):
  """Builds experience batches using an ExperienceCollector."""

  def __init__(self, experience_collector, run_inference, get_q_values,
               state_normalize_factor):

    self.experience_collector = experience_collector
    self.run_inference = run_inference
    self.get_q_values = get_q_values
    self.state_normalize_factor = state_normalize_factor


  def get_batches_stepwise(self):
    """Wraps get_batches(), keeping the current predictions constant for
    BATCHES_KEEP_CONSTANT steps.
    """

    cache = []

    for batches in self.get_batches():
      cache.append(batches)

      if len(cache) >= BATCHES_KEEP_CONSTANT:
        for cached_batches in cache:
          yield cached_batches
        cache = []


  def get_batches(self):
    """Yields randomized batches epsilon-greedy games.

    Maintains a replay memory at full capacity.
    """

    print("Initializing memory...")
    memory = ReplayMemory()
    while not memory.is_full():
      for experience in self.experience_collector.collect(play.random_strategy):
        memory.add(experience)

    memory.print_stats()

    for i in itertools.count():
      if i < START_DECREASE_EPSILON_GAMES:
        epsilon = 1.0
      else:
        epsilon = max(MIN_EPSILON,
                      1.0 - (i - START_DECREASE_EPSILON_GAMES) /
                      DECREASE_EPSILON_GAMES)

      strategy = play.make_epsilon_greedy_strategy(self.get_q_values, epsilon)

      for experience in self.experience_collector.collect(strategy):
        memory.add(experience)
        batch_experiences = memory.sample(BATCH_SIZE)
        yield self.experiences_to_batches(batch_experiences)


  def experiences_to_batches(self, experiences):
    """Computes state_batch, targets, actions."""

    batch_size = len(experiences)
    state_batch = np.zeros((batch_size, 16))
    next_state_batch = np.zeros((batch_size, 16))
    targets = np.zeros((batch_size,), dtype=np.float)
    actions = np.zeros((batch_size,), dtype=np.int)
    bad_action_batch = np.zeros((batch_size,), dtype=np.bool)
    available_actions_batch = np.zeros((batch_size, 4), dtype=np.bool)

    for i, experience in enumerate(experiences):
      state_batch[i, :] = (experience.state.flatten() *
                           self.state_normalize_factor)
      next_state_batch[i, :] = (experience.next_state.flatten() *
                                self.state_normalize_factor)
      actions[i] = experience.action
      bad_action_batch[i] = experience.game_over or experience.not_available
      available_actions_batch[i, experience.next_state_available_actions] = True

    good_action_batch = np.logical_not(bad_action_batch)

    targets[bad_action_batch] = -1
    targets[good_action_batch] = 0

    if GAMMA > 0:
      predictions = self.run_inference(next_state_batch)
      predictions[np.logical_not(available_actions_batch)] = -1
      max_qs = predictions.max(axis=1)
      max_qs = np.maximum(max_qs, -1)
      max_qs = np.minimum(max_qs, 0)
      targets[good_action_batch] += GAMMA * max_qs[good_action_batch]

    return state_batch, targets, actions
