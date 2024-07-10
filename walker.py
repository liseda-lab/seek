import abc
from typing import Any, List, Set, Tuple

import rdflib

from kg import KG

import numpy as np

class Sampler(metaclass=abc.ABCMeta):
    """Defines the Uniform Weight Weight sampling strategy.

    This sampling strategy is the most straight forward approach. With this
    strategy, strongly connected entities will have a higher influence on the
    resulting embeddings.

    Attributes:
        inverse: True if Inverse Uniform Weight sampling satrategy must be
            used, False otherwise. Default to False.

    """

    def __init__(self):
        pass

    def initialize(self) -> None:
        self.visited: Set[Any] = set()

    def sample_neighbor(self, kg: KG, walk, last):
        not_tag_neighbors = [
            x
            for x in kg.get_hops(walk[-1])
            if (x, len(walk)) not in self.visited
        ]

        if len(not_tag_neighbors) == 0:
            if len(walk) > 2:
                self.visited.add(((walk[-2], walk[-1]), len(walk) - 2))
            return None

        # Sample a random neighbor and add them to visited if needed.
        rand_ix = np.random.choice(range(len(not_tag_neighbors)))
        if last:
            self.visited.add((not_tag_neighbors[rand_ix], len(walk)))
        return not_tag_neighbors[rand_ix]


def extract_walks(graph, root, walks_per_graph, max_depth):

    sampler = Sampler()
    sampler.initialize()
    walks = []
    while len(walks) < walks_per_graph:
        new = (root,)
        d = 1
        while d // 2 < max_depth:
            last = d // 2 == max_depth - 1
            hop = sampler.sample_neighbor(graph, new, last)
            if hop is None:
                break
            new = new + (hop[0], hop[1])
            d = len(new) - 1
        walks.append(new)
    return list(set(walks))