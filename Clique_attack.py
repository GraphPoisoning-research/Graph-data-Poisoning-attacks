#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: 
Created Time: 
'''
import numpy as np
import networkx as nx
from functions import *


# 通过构建adversary nodes组成的dense community以提高模块度，
class Clique_attack(object):
    def __init__(self, G, epsilon, alpha):
        self.G = G
        self.size = self.G.number_of_nodes()
        self.epsilon = epsilon
        self.alpha = alpha
        self.advNodeList = adv_node_sample(self.G, self.alpha)


    def _clique_gen(self, adjMatrix):
        """
        Generates a new adjacency matrix by connecting nodes in the adversary node list
        to form a clique. A clique is a subset of vertices of an undirected graph such
        that every two distinct vertices are adjacent.

        :param adjMatrix: The original adjacency matrix representing the graph structure
            where rows and columns correspond to nodes.
        :type adjMatrix: numpy.ndarray
        :return: A modified adjacency matrix with a clique formed among the adversary nodes.
        :rtype: numpy.ndarray
        """
        modifyMatrix = adjMatrix.copy()
        advNodeList =  self.advNodeList
        for i in range(len(advNodeList)):    # adversary nodes内部构建Clique
            nodeI = advNodeList[i]
            for j in range(0, i):
                nodeJ = advNodeList[j]
                if nodeI != nodeJ:
                    modifyMatrix[nodeI, nodeJ] = 1
                    modifyMatrix[nodeJ, nodeI] = 1
        return modifyMatrix


    def _clique_gen_ipa(self, oriMatrix):
        """

        :param oriMatrix:
        :return:
        """
        modiMatrixIPA = self._clique_gen(oriMatrix)
        # 对嵌入攻击之后的邻接矩阵进行统一扰动
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        inverse_matrix = (np.random.rand(self.size, self.size) > p).astype(int)
        pertMatt = np.abs(modiMatrixIPA - inverse_matrix).astype(int)
        return pertMatt