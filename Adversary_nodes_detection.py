#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: 
Created Time: 
'''
from typing import List, Dict, Tuple
import numpy as np
import networkx as nx
import powerlaw
from typing import Optional


class AdversaryNodesDetection:
    """
    A class for identifying dishonest nodes based on link consistency.
    """

    def __init__(self, perturbedMatrix: np.ndarray):
        """
        Constructor.

        :param perturbedMatrix: An N x N NumPy matrix.
                              Matrix M[i, j] represents the link status (perturbed) 
                              between node i and node j as reported by node i.
                              This matrix is not required to be symmetric. Diagonal elements will be ignored.
        """
        if perturbedMatrix.shape[0] != perturbedMatrix.shape[1]:
            raise ValueError("The input perturbed matrix must be a square matrix (N x N).")

        self.reportedLinksMatrix = perturbedMatrix
        self.nodeCount = self.reportedLinksMatrix.shape[0]
        # _build_consistency_matrix will initialize self.consistencyMatrix
        self._build_consistency_matrix()

        # Initialize result attributes
        self.honestyScores: np.ndarray = np.array([])
        self.rankedNodes: List[Tuple[int, float]] = []
        self.identifiedAdversaries: np.ndarray = np.array([])

    def _build_consistency_matrix(self):
        """
        (Private method) Build an N x N consistency matrix based on the perturbed matrix.
        Matrix C[i, j] = 1 indicates that node i and j report consistently, otherwise 0.
        """
        if self.nodeCount == 0:
            self.consistencyMatrix = np.array([])
            return

        # print("Building link consistency matrix...")
        # Consistency matrix C[i,j] = 1 if M[i,j] == M[j,i]
        # (M == M.T) generates a boolean matrix, which we convert to integers (True=1, False=0)
        self.consistencyMatrix = (self.reportedLinksMatrix == self.reportedLinksMatrix.T).astype(int)
        # Diagonal elements are meaningless, set to 0
        np.fill_diagonal(self.consistencyMatrix, 0)
        # print("Consistency matrix construction completed.")

    def _calculate_honesty_scores(self):
        """
        (Private method) Calculate the honesty score H(i) for each node.
        """
        if self.nodeCount <= 1:
            return

        # print("Calculating honesty scores for all nodes...")
        # Sum each row of the consistency matrix, then divide by (N-1)
        sumOfConsistencies = np.sum(self.consistencyMatrix, axis=1)
        self.honestyScores = sumOfConsistencies / (self.nodeCount - 1)
        # print("Honesty score calculation completed.")

    def run_detection(self, numAdversaries: int = None, threshold: float = None):
        """
        Main method to run dishonest node detection.

        :param numAdversaries: int, optional. Number of adversaries to identify (hard threshold method).
        :param threshold: float, optional. Threshold for honesty scores (soft threshold method), 
                         nodes below this value are identified.
        """
        self._calculate_honesty_scores()
        if self.honestyScores.size == 0:
            print("Unable to calculate honesty scores.")
            return

        sortedIndices = np.argsort(self.honestyScores)
        self.rankedNodes = [(node_idx, self.honestyScores[node_idx]) for node_idx in sortedIndices]

        if numAdversaries is not None:
            # print(f"Using hard threshold method, identifying the {numAdversaries} nodes with lowest honesty scores...")
            self.identifiedAdversaries = sortedIndices[:numAdversaries]
        elif threshold is not None:
            # print(f"Using soft threshold method, identifying nodes with honesty scores below {threshold}...")
            adversary_indices = np.where(self.honestyScores < threshold)[0]
            self.identifiedAdversaries = adversary_indices
        else:
            print("Warning: No identification strategy provided (numAdversaries or threshold), unable to identify nodes.")

    def get_honesty_scores(self) -> np.ndarray:
        """Returns the honesty score array for all nodes."""
        return self.honestyScores

    def get_ranked_nodes(self) -> List[Tuple[int, float]]:
        """Returns a list of nodes sorted by honesty score from low to high, format: (node ID, score)."""
        return self.rankedNodes

    def get_adversaries(self) -> np.ndarray:
        """Returns the list of identified dishonest nodes."""
        return self.identifiedAdversaries