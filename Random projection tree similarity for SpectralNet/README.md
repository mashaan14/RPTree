### Files that we modified from the original SpectralNet code
- \core\pairs.py
	to implement our method
- \core\data.py
	to upload datasets
- \core\run.py
	

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
January 11, 2021.
