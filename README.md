# RPTree

[![DOI](http://img.shields.io/badge/doi-10.1109/ACCESS.2022.3195488-36648B.svg)](https://doi.org/10.1109/ACCESS.2022.3195488)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

## 	The Effect of Points Dispersion on the k-nn Search in Random Projection Forests
This is an implementation for the following paper:
```bibtex
@ARTICLE{9846977,
	author={Alshammari, Mashaan and Stavrakakis, John and Ahmed, Adel F. and Takatsuka, Masahiro},
	journal={IEEE Access}, 
	title={The Effect of Points Dispersion on the k-nn Search in Random Projection Forests}, 
	year={2022},
	volume={10},
	pages={80858-80868},
	doi={10.1109/ACCESS.2022.3195488}}
```

## How to use:

### RPTree.py
this python file contains the `BinaryTree` class, which performs the following tasks:
- `construct_tree`
- `get_leaf_nodes`
- `preorder_search`

### RunMain.py
this python file contains the driver code to create `BinaryTree` instance, then performs the following tasks:
- pick a random test sample, and remove it from the dataset.
- create the rpTree from the dataset.
- perform k-nn search on the rpTree to find the nearest neighbors for the test sample.

---
written by [Mashaan Alshammari](https://mashaan14.github.io/mashaan/)<br/>
August 01, 2022.
