# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 19:17:36 2021

@author: Mashaan
"""

import sklearn.preprocessing as preprocessing
import sklearn.datasets as datasets
from sklearn import random_projection
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt

import random
import RPTree

# small
# X = np.genfromtxt('sparse303_Instances.csv', delimiter=',')
# X = np.genfromtxt('ring238_Instances.csv', delimiter=',')
# X = np.genfromtxt('sparse622_Instances.csv', delimiter=',')
# X = np.genfromtxt('Agg788_Instances.csv', delimiter=',')

# mid
iris = datasets.load_iris()
X = iris['data']

# wine = datasets.load_wine()
# X = wine['data']

# BreastCancer = datasets.load_breast_cancer()
# X = BreastCancer['data']

# Digits = datasets.load_digits()
# X = Digits['data']

# large
# cal_housing = datasets.fetch_california_housing()
# X = cal_housing['data']

# X = np.genfromtxt('mGamma_Instances.csv', delimiter=',')
# X = np.genfromtxt('CreditCard_Instances.csv', delimiter=',')
# X = np.genfromtxt('CASP_Instances.csv', delimiter=',')

X = preprocessing.normalize(X)

Dataset = 'iris'
k = 5
NumOfRuns = 100
NumOfRows = X.shape[0]
NumOfCols = X.shape[1]

X_all = X
for whichProjection in ['2008_Dasgupta','2019_Yan','proposed','PCA']:
    for NumOfTrees in [1,2,3,4,5,10,20,40,60,80,100]:
        for run in range(NumOfRuns):
            # pick a random test sample, and remove it from the dataset
            X = X_all
            xIndex = random.randint(0,NumOfRows-2)
            X = np.delete(X, xIndex, 0)
            
            xDist = distance.cdist(X_all[xIndex:xIndex+1,:], X, 'euclidean')
            xDistSortIndex = xDist.argsort()[0,:k]
            xDistSort = xDist[0,xDistSortIndex]
            xNeighbors = X[xDistSortIndex,:]
                    
            for r in range(NumOfTrees):
                tree = RPTree.BinaryTree(X)
                tree_root = tree.construct_tree(tree, whichProjection) # '2008_Dasgupta' # '2008_Dasgupta' # '2019_Yan' #  'proposed' # 'PCA'
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
            print(whichProjection)
            print(xDistSort)
            print(xTreeDistSort)
            print('The algorithm miss rate = '+ str(TrueNeighborsMissed/k))
            print('The algorithm produces k distance = '+ str(xTreeDistSort[-1]-xDistSort[-1]))
            
            with open('Results-'+Dataset+'.csv', 'a') as my_file:
                # score, Number of trees, Method, Dataset, metric
                my_file.write('\n')
                my_file.write(str(np.round(TrueNeighborsMissed/k, 4))+','+str(NumOfTrees)+','+whichProjection+','+Dataset+',Avg. missing rate')
                my_file.write('\n')
                my_file.write(str(np.round(xTreeDistSort[-1]-xDistSort[-1], 4))+','+str(NumOfTrees)+','+whichProjection+','+Dataset+',Avg. distance error')
            
    #         if run==0:
    #             Results = np.array([(TrueNeighborsMissed/k), (xTreeDistSort[-1]-xDistSort[-1])])
    #         else:
    #             Results = np.vstack((Results, [(TrueNeighborsMissed/k), (xTreeDistSort[-1]-xDistSort[-1])]))
        
    #     if NumOfTrees == 1:
    #         ResultsAll = Results
    #     else:
    #         ResultsAll = np.hstack((ResultsAll,Results))
    # np.savetxt('Results-CASP-'+whichProjection+'.csv', ResultsAll, delimiter=",")
