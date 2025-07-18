#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: 
Created Time: 
'''
import numpy as np
import networkx as nx
import community as community_louvain
from sklearn import metrics
import matplotlib.pyplot as plt
import powerlaw
import networkit as nk
from networkit.nxadapter import nk2nx
from networkit.generators import ConfigurationModelGenerator


def result_read(readPath):
    resu = list()
    f = open(readPath, 'r')
    for line in f.readlines():
        line = line.strip('\n')
        resu.append(int(float(line)))
    return resu


def file_read(filepath):
    f = open(filepath, 'r')
    edges = []
    for line in f.readlines():
        line = line.strip().split()
        edges.append((int(line[0]), int(line[1])))
    return edges


def med_aver_compute(graph):

    medianDeg = list(dict(nx.degree(graph)).values())
    medianDeg.sort()
    medianDeg = medianDeg[int(len(graph.nodes())//2)]
    averageDeg = np.sum(list(dict(nx.degree(graph)).values()))/len(graph.nodes())
    return medianDeg, averageDeg


def med_aver_fromDegs(degs):

    medianDeg = degs
    medianDeg.sort()
    medianDeg = medianDeg[int(len(degs)//2)]
    averageDeg = np.sum(degs)/len(degs)
    return medianDeg, averageDeg


def clustering_coefficient(G):
    # average CC computation
    cc = nx.clustering(G)
    total = 0
    for i in range(len(cc)):
        total += cc[i]
    return total / len(cc)


def louvain_clustering(G, size):
    nodeList = list(G.nodes())
    for i in range(size):
        if i not in nodeList:
            G.add_node(i)
    partition = community_louvain.best_partition(G)
    resu = []
    for i in range(len(G.nodes())):
        resu.append([])
    for i in range(len(G.nodes())):
        resu[partition[i]].append(i)
    resu = [i for i in resu if i != []]
    return resu


def label_gen(re, n):
    label = np.zeros(n)
    for c in range(len(re)):
        for i in re[c]:
            label[i] = c
    return label


def shortest_path_length(G):

    return nx.average_shortest_path_length(G)


def modularity_compute(G, resu):
    """

    :param G:
    :param resu: result of clustering
    :return:
    """
    # RE: |Q-Q'|/Q
    return nx.algorithms.community.modularity(G, resu)


def ari_compute(label1, label2):
    # label1 should be the ground truth
    return metrics.adjusted_rand_score(label1, label2)


def ami_compute(label1, label2):

    return metrics.adjusted_mutual_info_score(label1, label2)


def gcc_compute(G):

    return nx.transitivity(G)/3


def Deg_distr_gen(G):
    degreeList = list(dict(nx.degree(G)).values())
    maxDeg = max(degreeList)
    degDistri = np.zeros(maxDeg + 1)
    for i in range(len(degDistri)):
        degDistri[i] = degreeList.count(i)
    return degDistri


def Deg_distr_fromDegs(degreeList):
    maxDeg = max(degreeList)
    degDistri = np.zeros(maxDeg + 1)
    for i in range(len(degDistri)):
        degDistri[i] = degreeList.count(i)
    return degDistri


def degree_dis_draw(oriDegdis, degDis):
    ind = np.arange(len(degDis))
    ind_ori = np.arange(len(oriDegdis))
    plt.plot(ind_ori, oriDegdis, color='green', alpha=1, linestyle='--', linewidth=2, marker='x', markersize='7',
             label='RABV-only')
    plt.plot(ind, degDis, color='blue', alpha=1, linestyle=':', linewidth=2, marker='s', markersize='7', label='LDPGen')
    # plt.plot(ind, LFGDPR, color='orange', alpha=1, linestyle='-.', linewidth=2, marker='^', markersize='7',
    #          label='LF-GDPR')
    # plt.plot(ind, RE, color='red', alpha=1, linewidth=2, marker='o', markersize='7', label='Wdt-SCAN')
    # plt.ylim(0, 1.0)
    # plt.xlim(1, 8)
    plt.show()
    return None


def line_3d(origin_degrees, eps1_degrees):
    # line
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('degree (in log2)')
    ax.set_zlabel('count')

    ax.plot(xs=np.log2(np.array(range(len(origin_degrees)))), ys=np.ones(len(origin_degrees)) * 1, zs=origin_degrees,
            c="black", label='Ground truth',  linewidth=2.0)

    ax.plot(xs=np.log2(np.array(range(len(eps1_degrees)))), ys=np.ones(len(eps1_degrees)) * 2, zs=eps1_degrees,
            c="blue", label='Block-HRG', linewidth=2.0)

    # ax.plot(xs=np.array(range(len(origin_degrees))), ys=np.ones(len(origin_degrees)) * 1, zs=origin_degrees,
    #         c="black", label='Ground truth', linewidth=2.0)
    #
    # ax.plot(xs=np.array(range(len(eps1_degrees))), ys=np.ones(len(eps1_degrees)) * 2, zs=eps1_degrees,
    #         c="blue", label='Block-HRG', linewidth=2.0)


    # ax.plot(xs=np.log10(np.array(range(len(santa_ldpgen_degrees)))), ys=np.ones(len(santa_ldpgen_degrees)) * 3,
    #         zs=santa_ldpgen_degrees,
    #         c="red", label='LDPGen', linewidth=2.0)
    # ax.plot(xs=np.log10(np.array(range(len(santa_lf_degrees)))), ys=np.ones(len(santa_lf_degrees)) * 4, zs=santa_lf_degrees,
    #         c="orange", label='LF-GDPR', linewidth=2.0)
    # ax.plot(xs=np.log10(np.array(range(len(santa_rr_degrees)))), ys=np.ones(len(santa_rr_degrees)) * 5,
    #         zs=santa_rr_degrees,
    #         c="green", label='RR', linewidth=2.0)

    # ax.axes.yaxis.set_visible(False)
    ax.axes.yaxis.set_ticklabels([])

    # ax.legend(loc='lower right')
    ax.legend(bbox_to_anchor=(1.0, 0.90))
    plt.show()


def adv_node_sample(G, alpha):
    """
    Samples a subset of nodes from a given graph using a specified proportion as the adversary nodes.

    This function takes a graph and a fraction (alpha) as input and returns a subset
    of nodes from the graph, chosen randomly without replacement. The number of nodes
    returned is approximately the total number of nodes in the graph multiplied by alpha.

    :param G: The input graph object, typically a networkx graph.
    :type G: networkx.Graph
    :param alpha: The proportion of nodes to sample from the graph. Should be a float
        value between 0 and 1.
    :type alpha: float
    :return: A numpy array containing the randomly selected subset of nodes.
    :rtype: numpy.ndarray
    """
    nodeList = list(G.nodes())
    nodeNum = len(nodeList)
    advNodeList = np.random.choice(nodeList, size=int(np.round(nodeNum*alpha)), replace=False)
    return advNodeList


def attacker_degree_dis_estimation(G, advNodeList):
    """
    Attacker estimates global node distribution based on partial node degree values combined with power-law distribution.

    This function performs the following steps:
    1. Extract degree values of captured attack nodes.
    2. Use powerlaw package to fit power-law distribution on these sample degree values,
       finding optimal gamma (alpha) and k_min through maximum likelihood estimation.
    3. Use the fitted power-law distribution model to generate a new degree sequence
       of the same scale as the original graph, as an estimate of the entire graph's degree distribution.

    :param G: networkx.Graph, complete graph object.
    :param advNodeList: numpy.ndarray, list of captured nodes.
    :return:
        - estimatedFullDegreeSequence (numpy.array): Estimated degree sequence of the entire graph.
        - fitResults (powerlaw.Fit): powerlaw fitting result object containing detailed information like gamma, k_min.
    """
    # Step 1: Extract sample node degree values
    allDegrees = dict(G.degree())
    advDegList = [allDegrees[node] for node in advNodeList]
    # Step 2: Use powerlaw package for power-law distribution fitting
    fitResults = powerlaw.Fit(advDegList, discrete=True, verbose=False)
    # Step 3: Use fitted model to generate final degree distribution estimate
    nTotal = G.number_of_nodes()
    # Generate random numbers from fitted power-law distribution object (fitResults.power_law)
    estimatedDegreesFloat = fitResults.power_law.generate_random(nTotal)
    # Round generated floating-point degree values and convert to integers, ensure degree values are at least 1
    estimatedFullDegreeSequence = np.round(estimatedDegreesFloat).astype(int)
    estimatedFullDegreeSequence[estimatedFullDegreeSequence < 1] = 1

    return estimatedFullDegreeSequence, fitResults


def improved_attacker_degree_dis_estimation(G, advNodeList):
    """
    Improved degree distribution estimation function, using hybrid generation method to reduce bias.
    """
    # Steps 1 and 2 same as before: extract sample degree values and perform fitting
    allDegrees = dict(G.degree())
    advDegList = [allDegrees[node] for node in advNodeList]
    fitResults = powerlaw.Fit(advDegList, discrete=True, verbose=False)
    # Step 3 (improved): Hybrid generation
    nTotal = G.number_of_nodes()
    nSampled = len(advDegList)
    nToGenerate = nTotal - nSampled  # Number of unknown node degree values to generate
    # Generate degree values only for unknown nodes
    generatedDegrees = fitResults.power_law.generate_random(nToGenerate)
    generatedDegrees = np.round(generatedDegrees).astype(int)
    generatedDegrees[generatedDegrees < 1] = 1
    # Step 4: Merge real samples with generated part
    # Convert advDegList to numpy array for concatenation
    sampledDegrees = np.array(advDegList)
    estimatedFullDegreeSequence = np.concatenate((sampledDegrees, generatedDegrees))
    # (Optional) Shuffle sequence to make it look more random
    np.random.shuffle(estimatedFullDegreeSequence)
    return estimatedFullDegreeSequence, fitResults


def attacker_degree_dis_estimation_resamp(G, advNodeList):
    """
    Generate global degree distribution using resampling method based on existing node degree values
    Use non-parametric resampling method, directly utilizing high-quality samples for global estimation.
    """
    # Step 1: Extract sample node degree values
    allDegrees = dict(G.degree())
    advDegList = [allDegrees[node] for node in advNodeList]
    # (Optional) Perform power-law fitting only to obtain analytical indicators (gamma, k_min, etc.), not for generation!
    fitResults = powerlaw.Fit(advDegList, discrete=True, verbose=False)
    # Step 2 (core): Non-parametric resampling
    nTotal = G.number_of_nodes()
    # Check if sample list is empty
    if not advDegList:
        return np.array([]), fitResults
    # Sample nTotal degree values with replacement from sample pool (advDegList)
    # This is the best estimate of global degree distribution
    estimatedFullDegreeSequence = np.random.choice(advDegList, size=nTotal, replace=True)
    return estimatedFullDegreeSequence, fitResults


def generate_proxy_graph_using_config_model(estimatedDegreeSequence):
    """
    BTER Generator module is not compatible with new python versions, here we use configuration model instead
    Use NetworKit's ConfigurationModelGenerator to create a proxy graph based on given degree sequence.
    This is the most robust and reliable proxy graph generation method.
    :param estimatedDegreeSequence: numpy.ndarray, estimated degree sequence for the entire graph.
    :return: networkx.Graph, generated proxy graph G' converted to networkx format.
    """
    if not nk or not ConfigurationModelGenerator:
        print("NetworKit package or its ConfigurationModelGenerator component unavailable, cannot generate proxy graph.")
        return None
    print("Generating proxy graph using NetworKit's Configuration Model...")
    try:
        # Convert degree sequence to integer list
        degreeSequence = estimatedDegreeSequence
        # Key check: Configuration model requires degree sequence sum to be even
        if sum(degreeSequence) % 2 != 0:
            # If odd, randomly select a node's degree +1 to balance, this is standard practice
            idx_to_increment = np.random.randint(0, len(degreeSequence))
            degreeSequence[idx_to_increment] += 1
            print("Warning: To meet graph construction requirements (even degree sum), one node's degree has been incremented by 1.")
        # 1. Create generator instance
        generator = ConfigurationModelGenerator(degreeSequence)
        # 2. Run generation algorithm
        nkGraph = generator.generate()
        # 3. Convert NetworKit graph to NetworkX graph
        proxyGraph = nk2nx(nkGraph)
        print("Proxy graph generation completed.")
        return proxyGraph
    except Exception as e:
        print(f"Error occurred during NetworKit graph generation: {e}")
        return None


def proxy_graph_gen_lfr(nodeCount, avgDeg, maxDeg, minCom, maxCom, mu):
    """
    BTER Generator module is not compatible with python 3.10, so we here use the LFR model instead.
    Generates a synthetic graph using the LFR (Lancichinetti-Fortunato-Radicchi) benchmark model.
    The method uses a power-law distribution for generating degree sequences and community size distributions
    and allows for specification of mixing parameters.
    :param nodeCount: The total number of nodes in the generated graph.
    :type nodeCount: int
    :param avgDeg: The average node degree for the degree distribution.
    :type avgDeg: int
    :param maxDeg: The maximum degree of nodes.
    :type maxDeg: int
    :param minCom: The minimum size of a community.
    :type minCom: int
    :param maxCom: The maximum size of a community.
    :type maxCom: int
    :param mu: The mixing parameter indicating the fraction of edges that a node shares with other communities.
               A value of 0 means all edges are internal, while a value of 1 means all edges are external.
    :type mu: float
    :return: A NetworkX graph object generated based on the specified LFR model parameters.
    :rtype: networkx.Graph
    """
    lfr_generator = nk.generators.LFRGenerator(n=nodeCount)
    # Set degree sequence - use positional parameters instead of keyword parameters
    lfr_generator.generatePowerlawDegreeSequence(avgDeg, maxDeg, -2)
    # Parameter order: averageDegree, maxDegree, nodeDegreeExp
    # Set community size sequence - also use positional parameters
    lfr_generator.generatePowerlawCommunitySizeSequence(minCom, maxCom, -1)
    # Parameter order: minCommunitySize, maxCommunitySize, communitySizeExp
    # Set mixing parameter
    lfr_generator.setMu(mu)
    # Generate graph
    # lfr_generator.generate()
    graph = lfr_generator.generate()
    print(f"Successfully generated graph! Nodes: {graph.numberOfNodes()}, Edges: {graph.numberOfEdges()}")
    graph = nk2nx(graph)
    return graph


def rr_pert(G, epsilon, size):
    # construct the edge list for each node
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    matt = np.zeros((size, size))
    edges = list(G.edges())
    for i in edges:
        matt[i[0], i[1]] = 1
        matt[i[1], i[0]] = 1
    inverse_matrix = (np.random.rand(size, size) > p).astype(int)
    pertMatt = np.abs(matt - inverse_matrix).astype(int)
    return pertMatt


def adjacency_matrix_to_graph(pertMatrix):
    """
    Generate NetworkX graph from adjacency matrix

    :param pertMatrix: Adjacency matrix (numpy array)
    :return: NetworkX graph object
    """
    # Create graph directly from numpy array
    G = nx.from_numpy_array(pertMatrix)
    return G


def graph_to_adjacency_matrix(G, size=None):
    """
    Generate adjacency matrix from NetworkX graph (inverse operation of above function)

    :param G: NetworkX graph object
    :param size: Specify matrix size (optional)
    :return: Adjacency matrix (numpy array)
    """
    if size is None:
        return nx.adjacency_matrix(G).toarray()
    else:
        # Ensure matrix size is specified value
        adj_matrix = np.zeros((size, size))
        edges = list(G.edges())
        for edge in edges:
            i, j = edge
            if i < size and j < size:  # Ensure indices are within range
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Undirected graph
        return adj_matrix


def ipa_edge_pert(advEdgeList, size, epsilon):
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
    newEdgeList = np.zeros((len(advEdgeList), size))
    p = np.exp(epsilon) / (1 + np.exp(epsilon))
    for i in range(len(advEdgeList)):
        inverse_edge = (np.random.rand(len(advEdgeList[i])) > p).astype(int)
        newEdgeList[i] = (np.abs((advEdgeList[i] - inverse_edge)).astype(int))
    return newEdgeList


def merge(pertMatt, advNodeList,advEdgeList):
    """
            First apply unified RR perturbation to all nodes, then directly replace
            adversary nodes' information in the noisy adjacency matrix as the final graph
            :param advNodeList: Adversary node list, length alpha*size
            :param advEdgeList: List of adjacency bit vectors submitted by adversary nodes, shape: (alpha*size)*size
            :return:
    """
    for i in range(advNodeList):
        pertMatt[advNodeList[i], :] = advEdgeList[i]
    return pertMatt


def estimate_adversary_count(G: nx.Graph, sensitivity: float = 1.0):
    """

    """

    if G.number_of_nodes() < 2:
        return 0, None

    degreeSequence = [d for _, d in G.degree()]

    try:
        fitResults = powerlaw.Fit(degreeSequence, discrete=True, verbose=False)
    except Exception as e:
        print(f"Powerlaw fit failed: {e}")
        return -1, None  # -1 means error

    ksDistance = fitResults.power_law.D
    print(f"K-S distance(D): {ksDistance:.4f}")

    # alpha_A ≈ sensitivity * D
    nodeCount = G.number_of_nodes()
    estimatedProportion = sensitivity * ksDistance

    if estimatedProportion > 1.0:
        estimatedProportion = 1.0

    estimatedCount = int(np.round(estimatedProportion * nodeCount))

    print(f"proportion of dishonest nodes: {estimatedProportion:.2%}")
    print(f"number of dishonest nodes:: {estimatedCount}")

    return estimatedCount, fitResults
