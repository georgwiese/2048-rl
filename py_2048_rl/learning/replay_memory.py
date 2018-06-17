"""Replay Memory"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import deque
import random

MEMORY_CAPACITY = 1e4

class ReplayMemory(object):
  """Keeps a set of Experiences in a Queue"""


  def __init__(self):
    self.queue = deque()


  def add(self, experience):
    """Add a single experience to queue."""

    self.queue.append(experience)
    if len(self.queue) > MEMORY_CAPACITY:
      self.queue.popleft()


  def print_stats(self):
    """Print memory stats."""

    total = len(self.queue)
    unavailable = len([1 for e in self.queue if e.not_available])
    lost = len([1 for e in self.queue if e.game_over])

    print("Memory stats:")
    print("  Experiences: ", total)
    print("  Unavailable: ", unavailable,
          "(%.1f%%)" % ((100 * unavailable / total),))
    print("  Lost       : ", lost, "(%.1f%%)" % ((100 * lost / total),))


  def is_full(self):
    """Return whether the memory is full."""

    return len(self.queue) >= MEMORY_CAPACITY


  def sample(self, count):
    """Returns a random sample of <count> experiences."""

    return random.sample(self.queue, count)
