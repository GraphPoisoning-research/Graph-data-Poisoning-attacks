#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: 
Created Time:
decreasing GCC
'''
import numpy as np
import networkx as nx
from functions import *



class Hub_spoke_attack(object):
    def __init__(self, G, epsilon, alpha):
        self.G = G
        self.size = self.G.number_of_nodes()
        self.epsilon = epsilon
        self.alpha = alpha
        self.advNodeList = adv_node_sample(self.G, self.alpha)


    def hub_spoke_gen(self, adjMatrix):
        """
        Generates a hub-and-spoke structure for adversary nodes by modifying the adjacency matrix.

        This function isolates all adversary nodes by severing their connections to other nodes.
        Among the adversary nodes, a hub node is selected, and all remaining adversary nodes
        are connected to the hub, resulting in a hub-and-spoke topology for these nodes.

        :param adjMatrix: The original adjacency matrix to be modified. The matrix represents
            the graph where each element indicates the connection between nodes.
        :type adjMatrix: numpy.ndarray
        :return: A modified adjacency matrix reflecting the hub-and-spoke structure for the
            adversary nodes while maintaining the rest of the graph's topology.
        :rtype: numpy.ndarray
        """
        modifyMatrix = adjMatrix.copy()
        advNodeList =  self.advNodeList
        for nodeI in advNodeList:    # Disconnect all edges with nodeI
            modifyMatrix[nodeI, :] = 0  
            # modifyMatrix[:, nodeI] = 0 
        # hubNode = np.random.choice(advNodeList)     # strategy 1：chose a node as Hub node randomly
            # strategy 2: mark the max degree node as Hub node
        degList = list(dict(self.G.degree()).values())
        advDegList = [degList[node] for node in advNodeList]
        maxNode = self.advNodeList[np.argmax(advDegList)]
        hubNode = maxNode
        # Construct triples, 2 strategies
        # for i in range(len(advNodeList)):    # strategy 1
        #     nodeI = advNodeList[i]
        #     if nodeI != hubNode:
        #         modifyMatrix[nodeI, hubNode] = 1
        #         modifyMatrix[hubNode, nodeI] = 1
        # strategy 2
        for i in range(self.size):
            nodeI = i
            modifyMatrix[hubNode, nodeI] = 1        # hub Node connect to all nodes 
            # modifyMatrix[nodeI, hubNode] = 1        # other nodes connect to hub Node
        modifyMatrix[hubNode, hubNode] = 0
        return modifyMatrix


    def hub_spoke_gen_ipa(self, oriMatrix):
        """

        :param oriMatrix:
        :return:
        """
        modiMatrixIPA = self.hub_spoke_gen(oriMatrix)
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        inverse_matrix = (np.random.rand(self.size, self.size) > p).astype(int)
        pertMatt = np.abs(modiMatrixIPA - inverse_matrix).astype(int)
        return pertMatt




