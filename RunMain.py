# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 19:17:36 2021

@author: Mashaan
"""

import sklearn.datasets as datasets
from sklearn import random_projection
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt

import random
import RPTree

# iris = datasets.load_iris()
# X = iris['data']

# wine = datasets.load_wine()
# X = wine['data']

# BreastCancer = datasets.load_breast_cancer()
# X = BreastCancer['data']

Digits = datasets.load_digits()
X = Digits['data']

# X = np.genfromtxt('sparse622_Instances.csv', delimiter=',')

k = 5
NumOfRuns = 100
NumOfRows = X.shape[0]
NumOfCols = X.shape[1]

X_all = X
for NumOfTrees in [1,2,3,4,5,10,20,40,60,80,100]:
    for run in range(NumOfRuns):
        # pick a random test sample, and remove it from the dataset
        X = X_all
        xIndex = random.randint(0,NumOfRows-2)
        X = np.delete(X, xIndex, 0)
        
        # identify the true k nearest neighbors using distance matrix
        xDist = distance.cdist(X_all[xIndex:xIndex+1,:], X, 'euclidean')
        xDistSortIndex = xDist.argsort()[0,:k]
        xDistSort = xDist[0,xDistSortIndex]
        xNeighbors = X[xDistSortIndex,:]
        
        for r in range(NumOfTrees):
            # create a tree object
            tree = RPTree.BinaryTree(X)
            # recursively split the tree
            tree_root = tree.construct_tree(tree)
            # traverse the test sample down the tree and return the neighbors in the leaf node
            xTree = tree.preorder_search(tree_root,RPTree.Node(X_all[xIndex,:]))
            if r==0:
                xForest = xTree
            else:
                xForest = np.concatenate((xForest, xTree), axis=0)
        
        xTree = np.unique(xForest, axis=0)
        xTreeDist = distance.cdist(X_all[xIndex:xIndex+1,:], xTree, 'euclidean')
        xTreeDistSortIndex = xTreeDist.argsort()[0,:k]
        xTreeDistSort = xTreeDist[0,xTreeDistSortIndex]
        xTreeNeighbors = xTree[xTreeDistSortIndex,:]
        
        # compute the distance between TrueNeighbors and TreeNeighbors to see if there is any missing neighbors
        TrueNeighbors_TreeNeighbors = distance.cdist(xNeighbors, xTreeNeighbors, 'euclidean')
        # scan the rows of the distance matrix, if there is a zero it means that neighbor has also been found by the tree
        TrueNeighborsFound = np.sum(np.count_nonzero(TrueNeighbors_TreeNeighbors==0, axis=1))
        TrueNeighborsMissed = k - TrueNeighborsFound
        
        print('NumberOftrees = '+ str(NumOfTrees) + ' , run = '+ str(run))
        print(xDistSort)
        print(xTreeDistSort)
        print('The algorithm miss rate = '+ str(TrueNeighborsMissed/k))
        print('The algorithm produces k distance = '+ str(xTreeDistSort[-1]-xDistSort[-1]))
        if run==0:
            Results = np.array([(TrueNeighborsMissed/k), (xTreeDistSort[-1]-xDistSort[-1])])
        else:
            Results = np.vstack((Results, [(TrueNeighborsMissed/k), (xTreeDistSort[-1]-xDistSort[-1])]))
    
    if NumOfTrees == 1:
        ResultsAll = Results
    else:
        ResultsAll = np.hstack((ResultsAll,Results))
np.savetxt("Results_Digits.csv", ResultsAll, delimiter=",")
