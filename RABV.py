#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author:
Created Time:
'''
import numpy as np
import networkx as nx



def file_read(filepath):
    f = open(filepath, 'r')
    edges = []
    for line in f.readlines():
        line = line.strip().split()
        edges.append((int(line[0]), int(line[1])))
    return edges


class RrGraph(object):
    def __init__(self, filepath, epsilon):
        self.filepath = filepath
        data = file_read(filepath)
        G = nx.Graph()
        G.add_edges_from(data)
        self.G = G
        self.size = self.G.number_of_nodes()
        self.epsilon = epsilon

    def pert(self):
        """

        :return:
        """
        # construct the edge list for each node
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        matt = np.zeros((self.size, self.size))

        edges = list(self.G.edges())
        for i in edges:
            matt[i[0], i[1]] = 1
            matt[i[1], i[0]] = 1

        inverse_matrix = (np.random.rand(self.size, self.size) > p).astype(int)
        pertMatt = np.abs(matt-inverse_matrix).astype(int)
        return pertMatt


    def ipa_edge_pert(self, advEdgeList):
        """
        Generates a new perturbed edge list by applying a differential privacy
        mechanism based on the given adversarial edge list. The perturbation
        process involves randomizing the edges using a privacy parameter.

        :param advEdgeList: List of edges where each edge is represented as
            an array. Represents the input adversarial edge lists.
        :type advEdgeList: numpy.ndarray
        :return: A numpy array containing the perturbed edge lists. Each entry
            corresponds to a randomized version of the input edge list based
            on the privacy parameter.
        :rtype: numpy.ndarray
        """
        newEdgeList = np.zeros((len(advEdgeList), self.size))
        p = np.exp(self.epsilon) / (1 + np.exp(self.epsilon))
        for i in range(len(advEdgeList)):
            inverse_edge = (np.random.rand(len(advEdgeList[i])) > p).astype(int)
            newEdgeList[i]=(np.abs((advEdgeList[i]-inverse_edge)).astype(int))
        return newEdgeList


    def merge(self, advNodeList,advEdgeList):
        """
        First apply unified RR perturbation to all nodes, then directly replace 
        adversary nodes' information in the noisy adjacency matrix as the final graph
        :param advNodeList: Adversary node list, length alpha*size
        :param advEdgeList: List of adjacency bit vectors submitted by adversary nodes, shape: (alpha*size)*size
        :return:
        """
        pertMatt = self.pert()
        for i in range(advNodeList):
            pertMatt[advNodeList[i], :] = advEdgeList[i]
        return pertMatt