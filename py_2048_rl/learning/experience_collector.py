"""Class that collects experience batches."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math

from py_2048_rl.game import play

# Parameters for undersampling
DO_UNDERSAMPLING = True
AVG_KEEP_PROB = 0.04
MIN_KEEP_PROB = 0.01

class ExperienceCollector(object):
  """Collects experiences by playing according to a particular strategy."""


  def get_keep_probability(self, index, length):
    """Computes the keep probability for the experience with a given index.

    First, the index is mapped to a value x between 0 and 1 (last index mapped
    to 0, index 0 mapped to 1). Then, the keep probability is computed by a
    function keep_prob = e^(ax) + MIN_KEEP_PROB, such that the average
    probability is AVG_KEEP_PROB.

    For small AVG_KEEP_PROB, a can be approximated by
    a = - 1 / (AVG_KEEP_PROB - MIN_KEEP_PROB).

    Args:
      index: zero-based index of the experience.
      length: total number of experiences.
    """
    if not DO_UNDERSAMPLING:
      return 1.0

    value = 1 - index / (length - 1)
    return (math.e ** (- 1 / (AVG_KEEP_PROB - MIN_KEEP_PROB) * value) +
            MIN_KEEP_PROB)


  def deduplicate(self, experiences):
    """Returns a new experience array that contains contains no duplicates."""

    state_set = set()
    filterted_experiences = []
    for experience in experiences:
      state_tuple = tuple(experience.state.flatten())
      if not state_tuple in state_set:
        state_set.add(state_tuple)
        filterted_experiences.append(experience)
    return filterted_experiences


  def collect(self, strategy, num_games=1):
    """Plays num_games random games, returns all collected experiences."""

    experiences = []
    for _ in range(num_games):
      _, new_experiences = play.play(strategy, allow_unavailable_action=False)
      deduplicated_experiences = self.deduplicate(new_experiences)
      count = len(deduplicated_experiences)
      experiences += [e for index, e in enumerate(deduplicated_experiences)
                      if (np.random.rand() <
                          self.get_keep_probability(index, count))]
    return experiences
