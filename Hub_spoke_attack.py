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


# 通过构建adversary nodes组成的dense community以提高模块度，
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
        for nodeI in advNodeList:    # 全体隔绝
            modifyMatrix[nodeI, :] = 0  # 断开从nodeI出发的所有边
            # modifyMatrix[:, nodeI] = 0  # 断开指向nodeI的所有边
        # 选择Hub node， 两种策略
        # hubNode = np.random.choice(advNodeList)     # 策略1：随机选择一个节点作为Hub node
            # 策略2：将最大度值节点作为Hub node
        degList = list(dict(self.G.degree()).values())
        advDegList = [degList[node] for node in advNodeList]
        maxNode = self.advNodeList[np.argmax(advDegList)]
        hubNode = maxNode
        # 构建连通三元组，两种策略
        # for i in range(len(advNodeList)):    # 策略1：adversary nodes内部所有节点连接到Hub node
        #     nodeI = advNodeList[i]
        #     if nodeI != hubNode:
        #         modifyMatrix[nodeI, hubNode] = 1
        #         modifyMatrix[hubNode, nodeI] = 1
        # 策略2： 所有节点（包含所有adversary nodes和普通nodes）都连接到hub node
        for i in range(self.size):
            nodeI = i
            modifyMatrix[hubNode, nodeI] = 1        # hub Node向所有节点连接
            # modifyMatrix[nodeI, hubNode] = 1        # 所有节点向hub Node连接
        modifyMatrix[hubNode, hubNode] = 0
        return modifyMatrix


    def hub_spoke_gen_ipa(self, oriMatrix):
        """

        :param oriMatrix:
        :return:
        """
        modiMatrixIPA = self.hub_spoke_gen(oriMatrix)
        # 对嵌入攻击之后的邻接矩阵进行统一扰动
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        inverse_matrix = (np.random.rand(self.size, self.size) > p).astype(int)
        pertMatt = np.abs(modiMatrixIPA - inverse_matrix).astype(int)
        return pertMatt




