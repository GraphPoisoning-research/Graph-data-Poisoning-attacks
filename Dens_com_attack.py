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
class Dens_com_attack(object):
    def __init__(self, G, epsilon, alpha):
        self.G = G
        self.size = self.G.number_of_nodes()
        self.epsilon = epsilon
        self.alpha = alpha
        self.advNodeList = adv_node_sample(self.G, self.alpha)


    def dens_community_gen(self, adjMatrix):
        """
        Generates a modified adjacency matrix by isolating adversary nodes and constructing a clique
        among them. Adversary nodes (specified in `advNodeList`) are first isolated by setting all
        connections from and to these nodes to zero, ensuring they are disconnected from the rest of the
        graph. Subsequently, a complete graph (clique) is created among the adversary nodes to foster
        maximum interconnection within this group.

        :param adjMatrix: The original adjacency matrix of the graph.
                          It's a 2D matrix where each entry represents the presence or absence of an edge
                          between a pair of nodes.
        :return: A modified adjacency matrix where adversary nodes are completely isolated from the
                 non-adversary nodes, and a clique is constructed among the adversary nodes.
        """
        modifyMatrix = adjMatrix.copy()
        advNodeList =  self.advNodeList
        for nodeI in advNodeList:    # 全体隔绝
            modifyMatrix[nodeI, :] = 0  # 断开从nodeI出发的所有边
            # modifyMatrix[:, nodeI] = 0  # 断开指向nodeI的所有边
        for i in range(len(advNodeList)):    # adversary nodes内部构建Clique
            nodeI = advNodeList[i]
            for j in range(0, i):
                nodeJ = advNodeList[j]
                if nodeI != nodeJ:
                    modifyMatrix[nodeI, nodeJ] = 1
                    modifyMatrix[nodeJ, nodeI] = 1
        return modifyMatrix


    def dens_community_gen_ipa(self, oriMatrix):
        """

        :param oriMatrix:
        :return:
        """
        modiMatrixIPA = self.dens_community_gen(oriMatrix)
        # 对嵌入攻击之后的邻接矩阵进行统一扰动
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        inverse_matrix = (np.random.rand(self.size, self.size) > p).astype(int)
        pertMatt = np.abs(modiMatrixIPA - inverse_matrix).astype(int)
        return pertMatt