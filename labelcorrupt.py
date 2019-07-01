from __future__ import print_function
from __future__ import division

import warnings
from math import modf
from numbers import Number
from itertools import cycle

import numpy as np
from sklearn.base import clone, BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels

__all__ = ['KnnLabelCorrupter', 'PercentLabelCorrupter', 'CorruptTrainLabelsClassifier']

class KnnLabelCorrupter(BaseEstimator, TransformerMixin):
    
    def __init__(self, corruption_size=0.01, noise_rate=0.1, random_state=None):
        self.corruption_size = corruption_size
        self.noise_rate = noise_rate
        self.random_state = random_state
    
    def fit_transform(self, X, y):
        ## Error checking of input and parameters
        # Verify inputs
        X, y = check_X_y(X, y, accept_sparse=True)
        classes = unique_labels(y)
        
        if len(classes) == 1:
            warnings.warn('Only 1 class given to fit so corruption not possible, returning without corrupting y.')
            return y        

        # Seed random number generator of sklearn
        generator = check_random_state(self.random_state)
        
        # Get the appropriate iterator where "cls" is the class
        # Used below when generating k in KNN
        def get_size_iterator(random_state):
            # Convert corruption_size to cycled iterator
            if isinstance(self.corruption_size, Number):
                size_iter = [self.corruption_size]
            elif callable(self.corruption_size):
                size_iter = self.corruption_size(random_state)
            return cycle(size_iter)  # Ensure we don't run out of k values
        
        # Check that noise_rate either is constant or has a value for each class value
        if isinstance(self.noise_rate, Number):
            noise_rate_dict = {c:self.noise_rate for c in classes}
        else:
            noise_rate_dict = self.noise_rate
        assert all(c in noise_rate_dict for c in classes)
        
        ## Corrupt y
        yc = y.copy()
        nbrs = NearestNeighbors()
        for c in classes:
            # Setup data and iterators
            classI = np.where(y==c)[0]
            nc = len(classI)
            Xc = X[classI,:]
            nbrs.fit(Xc)
            
            # Loop until corruption budget filled
            # Note: Elegant way to break a double loop from
            #  https://stackoverflow.com/questions/2597104/break-the-nested-double-loop-in-python
            budget = int(np.round(noise_rate_dict[c]*Xc.shape[0]))
            otherClasses = list(set(classes).difference([c]))
            corruptSet = set()

            # Select a random data point and a k from the generator
            rand_perm = generator.permutation(Xc.shape[0])
            size_iter = get_size_iterator(generator)
            for ii, (i, sz) in enumerate(cycle(zip(rand_perm, size_iter))):
                # Convert percent to k value
                if sz < 1 and sz >= 0:
                    sz = sz*Xc.shape[0] # Scale based on size of class

                # Handle non integers (expectation is sz)
                dec,k = modf(sz)
                if dec > 0 and generator.rand() < dec:
                    k += 1

                # Handle the case where k == 0
                if k == 0:
                    # If we have already looped through the whole dataset than start setting k = 1
                    if ii >= len(rand_perm):
                        k = 1
                    else:
                        continue

                # Correct k for training size and budget
                k = int(min(k, budget-len(corruptSet), Xc.shape[0]))
                xq = Xc[i,:]
                corruptClass = otherClasses[generator.randint(len(otherClasses))]

                # Get nearest neighbors and flip all to the corruptClass
                # xq = xq.reshape(1,-1)
                if len(xq.shape) < 2:
                    xq = xq.reshape(1,-1) # Handle the dense case where it is only an array
                _,nn = nbrs.set_params(n_neighbors = k).kneighbors(xq)
                yc[classI[nn[0]]] = corruptClass
                temp = len(corruptSet)
                corruptSet.update(nn[0])
                
                if len(corruptSet) >= budget:
                    break
                    
            # Check that corruption budget was met
            # (NOTE: I think this is already taken care of by handling the case k == 0 above
            #   but I've kept as a check just in case.)
            if(len(corruptSet) < budget):
                warnings.warn("Class " + str(c) + " did not meet its corruption budget.\n"
                              +"Probably due to k=0 at least sometimes.\n"
                              +"Corruption/Budget=%d/%d" % (len(corruptSet),budget))
        return yc

class PercentLabelCorrupter(KnnLabelCorrupter):
    def __init__(self, noise_rate=0.1, corrupt_chunk_perc=0.01, corrupt_chunk_perc_std=0.01/2, random_state=None):
        self.noise_rate = noise_rate
        self.corrupt_chunk_perc = corrupt_chunk_perc
        self.corrupt_chunk_perc_std = corrupt_chunk_perc_std
        self.random_state = random_state

    def fit_transform(self, *args, **kwargs):
        # Set corruption size for Knn Label Corrupter
        self.corruption_size = _create_perc_generator_func(
            perc_mean=self.corrupt_chunk_perc, 
            perc_std=self.corrupt_chunk_perc_std, 
            random_state=self.random_state)
        return super(PercentLabelCorrupter, self).fit_transform(*args, **kwargs)

class CorruptTrainLabelsClassifier(BaseEstimator, ClassifierMixin):
    # Setting corrupter_random_state to 'persistent'
    #  corrupts the same training data the same way
    #  This is important for using the same corruption when comparing models
    def __init__(self, estimator=None, corrupter=None, corrupter_random_state='persistent'):
        self.estimator = estimator
        self.corrupter = corrupter
        self.corrupter_random_state = corrupter_random_state
    
    # Simple corruption pre-process
    def fit(self, X, y, **kwargs):
        
        # Make a clone of estimator and corrupter
        self.random_state_ = self._check_corrupter_random_state(X)
        self.estimator_ = self._check_estimator()
        self.corrupter_ = self._check_corrupter()
        
        # Corrupt y labels
        y_corrupt = self.corrupter_.fit_transform(X, y)
        self.estimator_.fit(X, y_corrupt, **kwargs)
        return self

    def _check_corrupter_random_state(self, X):
        if self.corrupter_random_state == 'persistent':
            # Stupid but simple hash function (at least 1*1e6) of data
            # Only need the hash value to be different for like 10 data matrices
            random_state = int(np.sum(np.abs(X))/np.max(np.abs(X))/X.size*1e6)
        else:
            random_state = None
        return random_state

    def _check_estimator(self):
        estimator = self.estimator
        if estimator is None:
            return KNeighborsClassifier()
        else:
            return clone(estimator)

    def _check_corrupter(self):
        corrupter = self.corrupter
        if corrupter is None:
            return PercentLabelCorrupter(random_state=self.random_state_)
        else:
            return clone(corrupter).set_params(random_state=self.random_state_)
    
    # Expose all other attributes of estimator implicitly
    def __getattr__(self, attr):
        if attr == 'estimator_':
            raise AttributeError
        elif hasattr(self, 'estimator_') and hasattr(self.estimator_,attr):
            return getattr(self.estimator_,attr)
        else:
            raise AttributeError

## Auxiliary functions for creating a percent generator
def _perc_generator(perc_mean, perc_std, random_state=None):
    # Error check
    if perc_std < 0:
        raise ValueError('Percent standard deviation must be > 0')
    if perc_mean < 0 or perc_mean >= 1:
        raise ValueError('Percent mean must be in the interval [0, 1). ')
        
    # Handle special cases
    if perc_mean == 0:
        while True:
            yield 1
    if perc_std == 0:
        while True:
            yield perc_mean

    # Mean and variance parameterization of beta distribution
    # https://stats.stackexchange.com/questions/12232/
    #  calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
    perc_var = perc_std*perc_std
    alpha = ((1.0-perc_mean)/perc_var - 1.0/perc_mean)*perc_mean*perc_mean
    beta = alpha*(1.0/perc_mean - 1)
    
    # Sample from beta
    generator = check_random_state(random_state)
    while True:
        perc = generator.beta(alpha,beta)
        yield perc

# Wrap this special function in class so that string representation can be overridden
class _PercWrapper():
    def __init__(self, perc_mean, perc_std):
        self.perc_mean = perc_mean
        self.perc_std = perc_std
    def __call__(self, random_state=None):
        return _perc_generator(self.perc_mean, self.perc_std, random_state)
    def __str__(self):
        return '%.1f%%+/-%.1f%%' % (self.perc_mean*100, self.perc_std*100)
    def __repr__(self):
        return self.__str__()
    
def _create_perc_generator_func(perc_mean, perc_std, random_state=None):
    return _PercWrapper(perc_mean, perc_std)

# Meta selector estimator
# Originally used to make a meta estimator with GridSearchCV estimators as
#  the components and it just selects between estimators
# Thus it would have been GridSearchCV(MetaEstimator([GridSearchCV(Est1), GridSearchCV(Est2),...]))
# However, the cv_results_ of the inner estimators were not saved so this idea was abandoned
# If cv_results_ is not important, than this could be used to select the best GridSearchCV model
class MetaEstimator(BaseEstimator):
    def __init__(self, estimators, estimator_index=0):
        self.estimators = estimators
        self.estimator_index = estimator_index
        self.fitted_estimators_ = [None for _ in range(len(estimators))]
    
    def fit(self, X, y=None, **kwargs):
        self.fitted_estimators_[self.estimator_index] = self.estimators[self.estimator_index].fit(X,y,**kwargs)
        self.selected_estimator_ = self.fitted_estimators_[self.estimator_index]
        return self
    
    def score(self, X, y=None, **kwargs):
        return self.selected_estimator_.score(X, y, **kwargs)
    
    # Expose all other attributes of estimator implicitly
    def __getattr__(self, attr):
        if attr == 'selected_estimator_':
            raise AttributeError
        elif hasattr(self, 'selected_estimator_') and hasattr(self.selected_estimator_,attr):
            return getattr(self.selected_estimator_,attr)
        else:
            raise AttributeError

# Check estimators
if __name__ == '__main__':
    from sklearn.utils.estimator_checks import check_estimator
    check_estimator(KnnLabelCorrupter)
    try:
        check_estimator(LabelCorrupt)
    except AssertionError as err:
        if(str(err).startswith('0.73666666666666669 not greater than 0.83')):
            print('NOTE: Assertion error likely caused because default corruption noise_rate > 0, change to 0 to see if it passes the checks.')
            warnings.warn(str(err))
        else:
            raise

