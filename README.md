# Code for "Hyperparameter Selection under Localized Label Noise via Corrupt Validation"
If you use this code, please cite the following paper:

Hyperparameter Selection under Localized Label Noise via Corrupt Validation  
D. I. Inouye, P. Ravikumar, P. Das, A. Datta.  
*Learning with Limited Labeled Data (NeurIPS Workshop)*, 2017.  
https://www.davidinouye.com/publication/inouye-2017-hyperparameter/inouye-2017-hyperparameter.pdf  

A majority of this work (including this code) was completed by David I. Inouye during a summer internship 
at Rakuten Institute of Technology, Boston.

# Label Corruption and Quality Tools
This repository provides estimators related label corruption (or label noise).

One significant implication of label corruption is that model selection
evaluation is no longer standard because the evaluation measures are computed
on *corrupted* labels.

Thus, this project suggests an alternative model
selection procedure than the standard cross validation technique. At each fold,
the training data is synthetically corrupted (ideally, in a similar way to the original corruption
mechanism) while the testing data is kept the same.  This mimics the actual
goal which is to estimate a good model based only on corrupted training samples.

The labelcorrupt Python module provides a very strong empirical non-parametric corruption model via KNN. Note that
this model is dependent both on the data (i.e. dependent on X) and on nearby labels (i.e. y_i is dependent on y_j).

The labelquality module provides some estimators for predicting label quality based
on a probabilistic view of models. The labelquality module is untested and mostly shows
the basic idea of estimating label quality. Also, for text documents, we suggest
using a very simple probabilistic model such as Naive Bayes or KNN.

# Structure of repository
labelcorrupt.py - Module for label corruption via KNN.

labelquality.py - Module for estimating or correcting for label corruption via probabilistic estimators.

tests/ - Primarily includes a framework for performing synthetic experiments to validate
  the efficacy of the model selection methods.

tests/notebooks - Includes a two demos to show how the modules can be used. Demos use synthetic data.

# Requirements
Please use the `requirements.txt` file to create environment (merely used `conda list --export`).
Note that code was developed using older modules than the current ones.
In particular, the main module version numbers needed are below.

```
scikit-learn=0.19.2
pandas=0.20.3
plotly=2.7.0
jupyter=1.0.0
tornado==4.5.3
```

Conda installation commands from a blank Python 3 environment:

```
conda install scikit-learn=0.19
conda install pandas=0.20.3
conda install jupyter
conda install -c plotly plotly=2
conda install tornado==4.5.3
```
