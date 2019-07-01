import os
import sys
import time
import json
import io
import warnings

import numpy as np
import scipy.sparse
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder

# Import labelcorrupt module by appending a path
# This is used in other modules that import this module 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import labelcorrupt

# Simple timer class to display times between split calls
class ProgressTimer:
    _prevTime = 0
    _prefix = ''

    def __init__(self, prefix = ''):
        self._prefix = prefix
        self.start()
        pass

    def start(self):
        self._prevTime = time.time()

    def split(self, desc='(No description)'):
        curTime = time.time()
        splitTime = curTime - self._prevTime
        print('%8.2f s (%5.2f m) - %s%s' % (splitTime, splitTime/60, self._prefix, desc))
        sys.stdout.flush()
        self._prevTime = curTime

# Get unique row indices of a sparse matrix
# Similar to np.unique for dense matrices
def get_unique_indices_sparse(X):
    T = ProgressTimer('Unique rows in sparse X: ')
    assert np.all(X.data != 0)   # We assume that all data is non-zero
    nnz_row = np.array(np.sum(X != 0, axis=1)).transpose()[0]   # Get sparsity pattern
    exists_flag = np.array([0])   # To handle nnz_row[i] = 0
    unique_ind = []
    for i in range(0,int(1+np.max(nnz_row))):
        # Filter to indices with the same number of non-zeros
        cur_sel = nnz_row==i
        cur_ind = np.array([ind for ind,is_i in enumerate(cur_sel) if is_i], dtype=np.int)
        T.split('Number of rows with %d non-zero entries: %d' % (i,len(cur_ind)) )
        if len(cur_ind) == 0:
            continue
        # Handle the special case of i==0
        if i == 0:
            unique_ind.append(cur_ind[0]) # Just take the top
            continue

        # Get unique indices among those with the same number of non-zeros
        X_cur = X[cur_sel,:]
        X_indices = np.reshape(X_cur.indices, (len(cur_ind), i))
        X_data = np.reshape(X_cur.data, (len(cur_ind), i))
        X_temp = np.hstack((X_indices, X_data))
        internal_ind = np.unique(X_temp, axis=0, return_index=True)[1]
        unique_ind.extend(cur_ind[internal_ind])

    # In-place sort which returns None
    unique_ind.sort()
    return np.array(unique_ind)
