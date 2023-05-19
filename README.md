# motif-learning

An sklearn-like module for the unsupervised learning of motifs in time-series data.

#### Purpose of the Package
+ The purpose of the package is facilitate motif learning in time-series datasets, and provide a toolkit to study their compositional properties and visualise them. 

### Features
+ The Motif Learner class 
	- Fit on time-series
	- Motif visualisation
	- Motif compositional analysis
	- Hasse diagram of motifs
	- Motif pruning

### Getting Started
The package can (soon) be found on pypi hence you can install it using pip

#### Installation
```bash
pip install motif_learning
```

### Usage

#### Examples
```python

>>> from motif_learning import MotifLearner
>>> dataset = [1,2,3,0,6,1,2,3,0,3,1,2,3,6,6,2,5,3,1,2,3,0,1,2,1,2]
>>> motifl = MotifLearner(l_motif_range=[3,5])
>>> motifl.fit(dataset)
>>> motifl.get_motifs()
[[1, 2, 3], [1, 2, 3, 0]]
>>> motif_comp_mtx, pruned_motif_list = motifl.motif_composition_analysis()
>>> print(pruned_motif_list)
[[1, 2, 3, 0]]

```
For more detailed examples and visualisations, see `examples/`

### Contribution
Contributions are very welcome. 

Let us know if you notice a bug!

### Author
+ Main maintainer: Akshey Kumar
