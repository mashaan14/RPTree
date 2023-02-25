# RPTree

## 	The Effect of Points Dispersion on the k-nn Search in Random Projection Forests
This is an implementation for the following paper:
```

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
Provided by Mashaan Alshammari<br/>
mashaan14 at gmail dot com<br/>
mashaan dot awad at outlook dot com<br/>
August 01, 2022.