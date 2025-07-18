#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Author: 
Created Time: 
'''
from functions import *
from Dens_com_attack import Dens_com_attack





if __name__ == '__main__':
    file_enron = 'dataset/Email-Enron.txt'
    sizeEnron = 36692
    file_astro = 'dataset/CA-AstroPh-transform.txt'
    sizeAstro = 18772
    file_facebook = 'dataset/facebook_combined.txt'
    sizeFacebook = 4039
    file_santa = 'dataset/santa.txt'
    sizeSanta = 16216

    epsilon = 8
    alpha = 0.2

    filepath = file_facebook
    size = sizeFacebook

    print('Current graph: Facebook')

    G = nx.read_edgelist(filepath, nodetype=int)
    print('NX graph generated.')
    print('When epsilon = ', epsilon, 'alpha = ', alpha)
    pertMatrix = rr_pert(G, epsilon, size)
    print('RR perturbation done.')
    pertG = adjacency_matrix_to_graph(pertMatrix)
    print('Using louvain algorithm to cluster the perturbed graph.')
    pertClusteringResu = louvain_clustering(pertG, size)
    print('Clustering done.')
    pertModularity = modularity_compute(pertG, pertClusteringResu)

    print('Perturbed Modularity = ', pertModularity)

    modiMatrix = pertMatrix

    print('graph poisoning with Dense community attack......')
    densCom = Dens_com_attack(G, epsilon, alpha)
    # advNodeList = densCom.advNodeList
    print('poisoning done.')
    modiMatrix = densCom.dens_community_gen_ipa(modiMatrix)
    modified_G = adjacency_matrix_to_graph(modiMatrix)
    modified_ClusteringResu = louvain_clustering(modified_G, size)
    # modif_cluster_sizes = [len(cluster) for cluster in modified_ClusteringResu]
    modified_Modularity = modularity_compute(modified_G, modified_ClusteringResu)
    # print(modif_cluster_sizes)
    print('Poisoned Modularity = ', modified_Modularity)