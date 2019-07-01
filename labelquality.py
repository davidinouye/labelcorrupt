from __future__ import print_function
from __future__ import division

import warnings
from numbers import Number

import numpy as np
from scipy.sparse import issparse
import scipy.sparse as sparse

from sklearn.base import clone, BaseEstimator, ClassifierMixin, TransformerMixin, RegressorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import safe_indexing

from labelcorrupt import CorruptTrainLabelsClassifier

__all__ = ['ProbLabelQuality', 'MetaProbLabelCorruptClassifier', 'ProbLabelCorrector']

def _check_threshold(thresh):
    if isinstance(thresh, Number):
        return thresh
    else:
        raise ValueError('Threshold should be None or a number')

# Transforms X_in = zip(X,y) to X_out=quality_score
class ProbLabelQuality(BaseEstimator):
    def __init__(self, estimator, quality_scorer='classprob'):
        self.estimator = estimator
        self.quality_scorer = quality_scorer

    def fit(self, X, y=None, **fit_params):
        # Check and return quality_scorer
        def _check_quality_scorer(scorer):
            if callable(scorer):
                return scorer
            elif scorer == 'classprob':
                def class_prob(prob,y_index):
                    return [p[j] for p,j in zip(prob,y_index)]
                return class_prob
            else:
                raise ValueError('quality_scorer must be either \'classprob\''
                                 ' or a callable that takes two arguments of'
                                 ' probability matrix and y_index vector.')
        self.quality_scorer_ = _check_quality_scorer(self.quality_scorer)

        # Fit internal estimator
        estimator = clone(self.estimator)
        estimator.fit(X, y, **fit_params)
        self.internal_estimator_ = estimator 

        return self
        
    def predict(self, X, y=None, **predict_proba_kwargs):
        check_is_fitted(self,['internal_estimator_']) 

        # Get probabilities
        y_probs = self.internal_estimator_.predict_proba(X, **predict_proba_kwargs)

        # Map y to indices into y_probs
        label_to_index = {v:i for i,v in enumerate(self.internal_estimator_.classes_)}
        y_index = [label_to_index[label] for label in y]

        # Pass to quality scorer
        return np.array(self.quality_scorer_(y_probs, y_index))

    # Convenience method
    def fit_predict(self, X, y=None, predict_proba_params={}, **fit_params):
        return self.fit(X,y,**fit_params).predict(X,y,**predict_proba_params)

# Predicts whether a label is clean (y=0) or corrupted (y=1) based on threshold
class MetaProbLabelCorruptClassifier(ClassifierMixin,ProbLabelQuality):
    def __init__(self, estimator, quality_scorer='classprob', threshold=0.01, suppress_warning=False):
        self.estimator = estimator
        self.quality_scorer = quality_scorer
        self.threshold = threshold
        self.suppress_warning = suppress_warning

    def fit(self, X, y=None, **fit_params):
        # Validate input
        if not self.suppress_warning and y is not None:
            warnings.warn('This unsupervised *meta* classifier ignores y'
                          ' because it is trying to predict the meta/hidden'
                          ' value of whether this label was corrupted rather'
                          ' than trying to predict y itself. Please make sure you'
                          ' provide data as a list of (x,y) tuple pairs'
                          ': i.e. X_new = list(zip(X,y))  y is allowed for validation'
                          ' purposes such as cross validation.')
        assert len(X) > 0, 'Must have at least one value'

        # Extract internal X and y
        X_internal, y_internal = _extract_X_y(X)

        # Check threshold
        _check_threshold(self.threshold)
        quality_estimator = ProbLabelQuality(self.estimator, self.quality_scorer)
        quality_estimator.fit(X_internal, y_internal, **fit_params)
        self.quality_estimator_ = quality_estimator
        return self

    # Give unthresholded quality scores as decision function
    def decision_function(self, X, **predict_proba_kwargs):
        check_is_fitted(self,['quality_estimator_']) 
        assert len(X) > 0, 'Must have at least one value'

        # Extract internal X and y
        X_internal, y_internal = _extract_X_y(X)
        return np.array([1-x for x in self.quality_estimator_.predict(X_internal, y_internal, **predict_proba_kwargs)])

    # Give prediction by thresholding decision function score
    def predict(self, X, **predict_proba_kwargs):
        check_is_fitted(self,['quality_estimator_']) 
        y_score = self.decision_function(X, **predict_proba_kwargs)
        return (np.array(y_score) > self.threshold).astype(np.int)

# Extract internal X and y from X
def _extract_X_y(xy_tuples):
    # TODO Handle X for other cases like pandas
    X = [xy[0] for xy in xy_tuples]
    y = [xy[1] for xy in xy_tuples] 
    if issparse(X[0]):
        X = sparse.vstack(X, format='csr')
    else:
        X = np.vstack(X)
    return (X, y)

class ProbLabelCorrector(ProbLabelQuality, TransformerMixin):
    def __init__(self, estimator, quality_scorer='classprob', pred_threshold=0.01):
        self.estimator = estimator
        self.quality_scorer = quality_scorer
        self.pred_threshold = pred_threshold # The threshold to choose predicted over original

    def fit(self, X, y=None, **fit_params):
        # Check threshold
        _check_threshold(self.pred_threshold)
        quality_estimator = ProbLabelQuality(self.estimator, self.quality_scorer) 
        quality_estimator.fit(X,y,**fit_params)
        self.quality_estimator_ = quality_estimator
        return self

    def transform(self, X, y=None, **predict_proba_kwargs):
        check_is_fitted(self,['quality_estimator_']) 

        # Get quality score
        quality_score = self.quality_estimator_.predict(X, y, **predict_proba_kwargs)

        # Get newly predicted labels
        y_pred = self.quality_estimator_.internal_estimator_.predict(X)

        # Choose between predicted and original labels based on quality estimate
        y_cleaned = [(orig_label if score > self.pred_threshold else pred_label) 
                     for score, orig_label, pred_label in zip(quality_score,y,y_pred)]

        return y_cleaned

    def fit_transform(self, X, y, predict_proba_kwargs={}, **fit_params):
        self.fit(X, y, **fit_params)
        return self.transform(X, y, **predict_proba_kwargs)

