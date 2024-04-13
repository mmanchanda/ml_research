import grakel as gr
import networkx as nx
import numpy as np
import unittest


class CosineDegreeSimilarity(gr.Kernel):
    """
    Computes the cosine similarity between graphs based on their degree distributions.
    """

    def __init__(self, normalize=False, verbose=False):
        super().__init__(normalize=normalize, verbose=verbose)

    def graph_feature_vector(self, graph):
        """
        Extracts degree features efficiently for a Grakel-compatible graph.

        Args:
            graph: Grakel-compatible graph representation.

        Returns:
            numpy.ndarray: Vector of node degrees.
        """
        return np.array([(len(list(graph.neighbors(node)))) for node in graph.nodes()])

    def pairwise_operation(self, graph1, graph2):
        """
        Calculates the cosine similarity between two graphs' feature vectors.

        Args:
            graph1: Grakel-compatible graph representation.
            graph2: Grakel-compatible graph representation.

        Returns:
            float: Cosine similarity between the graphs.
        """
        features1 = self.graph_feature_vector(graph1)
        features2 = self.graph_feature_vector(graph2)
        return np.dot(features1, features2) / (
            np.linalg.norm(features1) * np.linalg.norm(features2) + 1e-10)  # Adding a small constant for numerical stability, in case 0
        


class TestCosineDegreeSimilarity(unittest.TestCase):
    def setUp(self):
        self.graph1 = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
        self.graph2 = nx.DiGraph([(1, 2), (2, 3)])

    def test_identical_graphs(self):
        kernel = CosineDegreeSimilarity()
        similarity = kernel.pairwise_operation(self.graph1, self.graph1)
        self.assertAlmostEqual(similarity, 1.0,2)

    def test_dissimilar_graphs(self):
        kernel = CosineDegreeSimilarity()
        similarity = kernel.pairwise_operation(self.graph1, self.graph2)
        self.assertAlmostEqual(similarity, 0.8164965, places=2)
                    
if __name__ == "__main__":
    unittest.main()


