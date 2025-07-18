#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: 
Created Time: 
'''
from functions import *


# Build a dense community of adversary nodes to improve modularity
class Bridge_build_attack(object):
    def __init__(self, G, epsilon, alpha):
        self.G = G
        self.epsilon = epsilon
        self.size = self.G.number_of_nodes()
        self.alpha = alpha
        self.advNodeList = adv_node_sample(self.G, self.alpha)

    def bridge_construct(self, adjMatrix):
        """
        Without knowing community distribution, let adversary nodes randomly build 
        bridges to random nodes. How many edges? Keep original degree values unchanged?
        :return:
        """
        modiMatrix = adjMatrix.copy()
        advNodeList = self.advNodeList
        degList = list(dict(self.G.degree()).values())
        advDegList = [degList[node] for node in advNodeList]
        for nodeI in advNodeList:    # Isolate all
            modiMatrix[nodeI, :] = 0  # Disconnect all edges from nodeI
            # modiMatrix[:, nodeI] = 0  # Disconnect all edges to nodeI, this step is not allowed since the attacker only controlling adversary nodes

        # Re-establish random connections for each adversary node
        for i, nodeI in enumerate(advNodeList):
            candidate_nodes = list(range(self.size))
            candidate_nodes.remove(nodeI)
            # Randomly shuffle candidate nodes and select nodes for connection
            np.random.shuffle(candidate_nodes)
            # for j in range(advDegList[i]):
            for j in range(max(advDegList[i], 200)):     # Avoid ineffectiveness for low-degree nodes, make each node connect to at least 200 other nodes
                nodeJ = candidate_nodes[j]
                modiMatrix[nodeI, nodeJ] = 1
                # modiMatrix[nodeJ, nodeI] = 1
        return modiMatrix


    def bridge_construct_ipa(self, oriMatrix):
        """
        This function generates a perturbed matrix based on an initially modified adjacency
        matrix and applies a probabilistic transformation using the parameter epsilon.
        The perturbation process involves calculating a random inverse matrix and altering
        the values in the modified adjacency matrix accordingly to generate the final
        perturbed matrix.

        :param oriMatrix: Original adjacency matrix that serves as input for constructing
            an intermediate modified matrix (of type numpy.ndarray)
        :return: A perturbed adjacency matrix after applying probabilistic perturbation
            and inverse matrix calculations (of type numpy.ndarray)
        """
        modiMatrixIPA = self.bridge_construct(oriMatrix)
        # Apply unified perturbation to the adjacency matrix after embedding attack
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        inverse_matrix = (np.random.rand(self.size, self.size) > p).astype(int)
        pertMatt = np.abs(modiMatrixIPA - inverse_matrix).astype(int)
        return pertMatt