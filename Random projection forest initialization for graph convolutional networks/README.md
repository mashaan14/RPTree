### Files that we modified from the original [GCN](https://github.com/tkipf/gcn) code
- \GCN\RPTree.py
	a code that returns an rpTree given a feature matrix X
- \GCN\utils.py
	in line 99 we added a function called `load_data_rpForest` that returns an adjacency matrix based on rpForest
- \GCN\train.py
	in line 29 we called the function `utils.load_data_rpForest` to work on adjacency matrix based on rpForest
	

### Files that we modified from the original [LDS](https://github.com/lucfra/LDS-GNN) code
- \LDS\RPTree.py
	a code that returns an rpTree given a feature matrix X
- \LDS\lds.py
	in line 302 we added a code that returns an adjacency matrix based on rpForest
- \LDS\hyperparams.py
	in line 177 we added a code that randomly picks a percentage of edges that were missed by rpForest
	

## Python Setup

- Download and install Anaconda Navigator
- launch Anaconda Prompt to execute the following commands:
	- `conda create -n tf15 python tensorflow=1.15`
	- `conda activate tf15`
	- `conda remove --force tensorflow-estimator`
	- `conda install -c anaconda tensorflow-estimator==1.15.1`
	- `conda install -c anaconda scikit-learn`
	- `conda install -c conda-forge munkres`
	- `conda install -c conda-forge python-annoy`
	- `conda install -c conda-forge keras==2.3.1`
	- `conda install -c anaconda spyder`

- To start working in this enviroment, launch Anaconda Prompt and type:
	- `activate tf15`
	- `spyder`
---
written by Mashaan Alshammari<br/>
mashaan14 at gmail dot com<br/>
mashaan dot awad at outlook dot com<br/>
April 23, 2022.
