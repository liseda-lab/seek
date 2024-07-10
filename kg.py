import itertools
from collections import defaultdict
from typing import List, Set, Tuple
import rdflib


class Vertex(object):
    """Represents a Vertice in the Knowledge graph."""
    vertex_counter = itertools.count()

    def __init__(self, name, predicate=False, vprev=None, vnext=None):
        self.name = name
        self.predicate = predicate
        self.vprev = vprev
        self.vnext = vnext
        self.id = next(self.vertex_counter)

    def __eq__(self, other):
        if other is None:
            return False
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        if self.predicate:
            return hash((self.id, self.vprev, self.vnext, self.name))
        return hash(self.name)

    def __lt__(self, other):
        return self.name < other.name

    def __str__(self):
        return self.name


class KG(object):
    """Represents a Knowledge Graph."""

    def __init__(
        self,
        location=None,
        file_type=None,
        label_predicates=None,
        is_remote=False,):
        
        self.file_type = file_type
        if label_predicates is None:
            self.label_predicates = []
        else:
            self.label_predicates = label_predicates
        self.location = location
        self.is_remote = is_remote

        self._inv_transition_matrix = defaultdict(set)
        self._transition_matrix = defaultdict(set)
        self._vertices = set()
        self._entities = set()

    def _get_rhops(self, vertex: str) -> List[Tuple[str, str]]:
        """Returns a hop (vertex -> predicate -> object)"""
        if isinstance(vertex, rdflib.term.URIRef):
            vertex = Vertex(str(vertex))  # type: ignore
        elif isinstance(vertex, str):
            vertex = Vertex(vertex)  # type: ignore
        hops = []

        predicates = self._transition_matrix[vertex]
        for pred in predicates:
            assert len(self._transition_matrix[pred]) == 1
            for obj in self._transition_matrix[pred]:
                hops.append((pred, obj))
        return hops

    def add_vertex(self, vertex: Vertex) -> None:
        """Adds a vertex to the Knowledge Graph."""
        self._vertices.add(vertex)
        if not vertex.predicate:
            self._entities.add(vertex)

    def add_edge(self, v1: Vertex, v2: Vertex) -> None:
        """Adds a uni-directional edge."""
        self._transition_matrix[v1].add(v2)
        self._inv_transition_matrix[v2].add(v1)

    def get_hops(self, vertex: str) -> List[Tuple[str, str]]:
        return self._get_rhops(vertex)

    def get_inv_neighbors(self, vertex: Vertex) -> Set[Vertex]:
        """Gets the reverse neighbors of a vertex."""
        if isinstance(vertex, str):
            vertex = Vertex(vertex)
        return self._inv_transition_matrix[vertex]

    def get_neighbors(self, vertex: Vertex) -> Set[Vertex]:
        """Gets the neighbors of a vertex."""
        if isinstance(vertex, str):
            vertex = Vertex(vertex)
        return self._transition_matrix[vertex]

    def remove_edge(self, v1: str, v2: str) -> None:
        """Removes the edge (v1 -> v2) if present."""
        if v2 in self._transition_matrix[v1]:
            self._transition_matrix[v1].remove(v2)


def rdflib_to_kg(rdflib_g, label_predicates=[]):
    """ Transforms an rdflib graph into KG."""
    kg = KG()
    for (s, p, o) in rdflib_g:
        if p not in label_predicates:
            s_v = Vertex(str(s))
            o_v = Vertex(str(o))
            p_v = Vertex(str(p), predicate=True, vprev=s_v, vnext=o_v)
            kg.add_vertex(s_v)
            kg.add_vertex(p_v)
            kg.add_vertex(o_v)
            kg.add_edge(s_v, p_v)
            kg.add_edge(p_v, o_v)
    return kg
