import time
import warnings
import multiprocessing
from numbers import Number
from itertools import cycle, islice, repeat

import numpy as np
from scipy import stats
import pandas as pd

import plotly.offline as py
import plotly.graph_objs as go

from sklearn.base import clone, BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import check_cv
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing

# Shared files for testing
from shared import labelcorrupt
from shared import ProgressTimer
from labelcorrupt import PercentLabelCorrupter, CorruptTrainLabelsClassifier

# To allow clean test in grid search cv
from gridsearchcv_clean_test_mod_0_19 import _GridSearchCV_CleanTest

# Just used to simplify passing multiple datasets to functions
# . e.g. X_train, X_test, y_train, y_train_clean, y_test_clean ...
class DataContainer(object):
    pass

# Make simple datasets (in particular the double moons one)
def make_dataset(name, n_samples, random_state=None):
    ## Load a dataset
    if name == 'double_moons':
        X, y = make_moons(n_samples=int(n_samples), noise=0.2, random_state=random_state)
        idx = list(range(int(1/4*n_samples)))
        X[idx,0] += 3
        X[idx,1] += 1
        y[idx] += 2
        # Shuffle dataset
        shuffle_ind = np.random.permutation(X.shape[0])
        X = X[shuffle_ind,:]
        y = y[shuffle_ind]
    elif name == 'super_simple':
        X, y = make_super_simple(n_samples=n_samples, separation=3, random_state=random_state)
    else:
        X, y = make_moons(n_samples=int(n_samples/2), noise=0.2, random_state=random_state)
    return (X, y)


# Creates an DataContainer D with corrupted y
#  and the clean y as D.y_clean
def create_corrupt_dataset(X, y,
                    noise_rate = 0.2,
                    corrupt_chunk_perc = 0.01,
                    corrupt_chunk_perc_std = 0.005,
                    random_state=None,
                    **kwargs):
    # Set random state
    prev_state = np.random.get_state()
    gen = check_random_state(random_state)
    np.random.set_state(gen.get_state())
    
    # Setup dataset
    D = DataContainer()
    D.X, D.y_clean = (X, y)

    # Corrupt labels
    D.y = PercentLabelCorrupter(
        noise_rate=noise_rate,
        corrupt_chunk_perc=corrupt_chunk_perc,
        corrupt_chunk_perc_std=corrupt_chunk_perc_std,
        random_state=random_state,
    ).fit_transform(D.X, D.y_clean)
    
    # Reset random state
    np.random.set_state(prev_state)
    return D

# Wrap the base estimator in a CV estimator
#  with CorruptTrainLabelsClassifier with given corruption parameters
def corrupt_cv(cv_estimator,
        noise_rate = 0.2,
        corrupt_chunk_perc = 0.01, # Approximate percentage to flip together
        corrupt_chunk_perc_std = 0.005, # Variation around it
        ):
    # Clone estimator
    cv_corrupt = clone(cv_estimator)

    # Wrap estimator in a label corrupter
    cv_corrupt.estimator = CorruptTrainLabelsClassifier(
        estimator=cv_corrupt.estimator,
        corrupter=PercentLabelCorrupter(
            noise_rate=noise_rate,
            corrupt_chunk_perc=min(1-1e-6, corrupt_chunk_perc),
            corrupt_chunk_perc_std=corrupt_chunk_perc_std,
        )
    )

    # Convert parameters to nested parameters
    cv_corrupt.param_grid = {'estimator__%s' % k: v for k,v in cv_corrupt.param_grid.items()}

    return cv_corrupt

# Replace CV estimator with a specially modified CV estimator
#  which trains on the given data but tests on y_clean (which is 
#  passed as a parameter to the fit method of _GridSearchCV_CleanTest
def clean_test(cv_estimator):
    clean_params = clone(cv_estimator).get_params(deep=False)
    cv_clean_test =_GridSearchCV_CleanTest(**clean_params)
    return cv_clean_test

# Main synthetic experiment for label corruption that runs a 
#  nested CV model selection with different model selection 
#  methodologies
def synthetic_corruption_experiment(
        # Main parameters
        X, y,
        estimator_configs,
        # Experimental parameters
        n_cv_splits_outer = 3,
        n_cv_splits_inner = 3,
        n_corrupted_datasets = 3,
        # Corruption parameters
        noise_rate=0.2,
        corrupt_chunk_perc=0.02, # Approximate percentage to flip together
        corrupt_chunk_perc_std=0.02/2, # Variation around it
        n_jobs=1,
        meta_model_selection=True,
        cv_verbose=0,
    ):
    # Setup
    T_top = ProgressTimer()
    if len(estimator_configs) == 1 and meta_model_selection:
        meta_model_selection = False
        warnings.warn('Changing meta_model_selection to False since only one estimator config')
    corrupt_params = dict(
        noise_rate=noise_rate,
        corrupt_chunk_perc=corrupt_chunk_perc,
        corrupt_chunk_perc_std=corrupt_chunk_perc_std,
    )
    
    # Generate splits and corrupted labels
    D_arr = [create_corrupt_dataset(X, y, random_state=i, **corrupt_params)
             for i in range(n_corrupted_datasets)]
    T_top.split('Create corrupted datasets')
    
    # Setup different model selection methodologies
    methodology = [
        dict(
            name='CV',
            data_params_func=lambda D, ind, isTest=False: {'X':safe_indexing(D.X, ind), 'y':safe_indexing(D.y, ind)},
            cv_wrapper=lambda cv_est: cv_est,
        ),
        dict(
            name='Corrupt CV (r=%.0f%%,sz=%.1f+/-%.1f%%)' % (
                100*noise_rate,
                100*corrupt_chunk_perc,
                100*corrupt_chunk_perc_std,
            ),
            data_params_func=lambda D, ind, isTest=False: {'X':safe_indexing(D.X, ind), 'y':safe_indexing(D.y, ind)},
            cv_wrapper=lambda cv_est, cp=corrupt_params: corrupt_cv(cv_est, **cp),
        ),
        dict(
            name='(Oracle) CV with Clean Test',
            data_params_func= lambda D, ind, isTest=False: (
                {'X':safe_indexing(D.X, ind), 'y':safe_indexing(D.y, ind), 'y_clean':safe_indexing(D.y_clean, ind)}
                if not isTest
                else {'X':safe_indexing(D.X, ind), 'y':safe_indexing(D.y_clean, ind)}
            ),
            cv_wrapper=lambda cv_est: clean_test(cv_est),
        ),
        dict(
            name='(Oracle) CV with Clean Train & Test', 
            data_params_func=lambda D, ind, isTest=False: {'X':safe_indexing(D.X, ind), 'y':safe_indexing(D.y_clean, ind)},
            cv_wrapper=lambda cv_est: cv_est,
        ),
    ]
    # So that we can limit to only 1 methodology
    fit_order = [2,0,1,3]
    #fit_order = [2] # Use when trying to find rough bounds on parameter (reduces time by 4x)
    if len(fit_order) < 4:
        warnings.warn('Not fitting all 4 methodologies')

    # Run estimators
    results = pd.DataFrame(None, columns=[
        'Methodology','Synthetic Corruption Index','Outer Split','Inner Split',
        'Estimator','Estimator Label','Params','Params Label','Params Index','Score','Clean Score',
    ])
    clean_cv_models = []
    T_m = ProgressTimer('  ')
    for fi in fit_order:
        data_params_func = methodology[fi]['data_params_func']
        cv_wrapper = methodology[fi]['cv_wrapper']
        T_d = ProgressTimer('    ')
        for di, D in enumerate(D_arr):
            # Setup default values for all results in this section
            data_dict={
                'Methodology': methodology[fi]['name'],
                'Methodology Index': fi,
                'Synthetic Corruption Index': di
            }
            
            # Create stratified splits based on *corrupted* y rather than clean y
            if n_cv_splits_outer == 1:
                outer_splits = [next(check_cv(cv=10, classifier=True).split(np.zeros(D.y.shape), D.y))]
            else:
                outer_splits = list(check_cv(cv=n_cv_splits_outer, classifier=True).split(np.zeros(D.y.shape), D.y))
            T_s = ProgressTimer('      ')
            for si, (train, test) in enumerate(outer_splits):
                # Default dict for this outer slit
                split_dict = data_dict.copy()
                split_dict.update({'Outer Split': si})

                # Train all CV models
                if n_cv_splits_inner == 1:
                    y_temp = data_params_func(D,train)['y']
                    inner_cv = [next(check_cv(cv=10, classifier=True).split(np.zeros(y_temp.shape), y_temp))]
                else:
                    inner_cv = n_cv_splits_inner
                cv_inner = {'pre_dispatch': 'n_jobs', 'n_jobs': n_jobs, 'cv': inner_cv, 'refit': False, 'verbose': cv_verbose, 'return_train_score': False}
                cv_models = [
                    # Wrap model based on methodology (e.g. add corruption)
                    cv_wrapper(
                        GridSearchCV(
                            estimator=config['estimator'],
                            param_grid=config['cv_params'],
                            **cv_inner,
                        )
                    ).fit(
                        # Fit with certain X,y and (possibly) y_clean based on methodology
                        #  (e.g. Append y_clean=.. for (Oracle) CV with Clean Test)
                        **data_params_func(D,train)
                    )
                    for config in estimator_configs
                ]

                # Save clean models to extract clean CV scores of other models
                if methodology[fi]['name'] == '(Oracle) CV with Clean Test':
                    if len(clean_cv_models) < len(D_arr):
                        clean_cv_models = [[] for _ in D_arr]
                    if len(clean_cv_models[di]) < len(outer_splits):
                        clean_cv_models[di] = [None for _ in outer_splits]
                    clean_cv_models[di][si] = cv_models

                # Functions to unwrap CorruptTrainLabelsClassifier wrapper
                def unwrap_corrupt(est):
                    return est.estimator if isinstance(est,CorruptTrainLabelsClassifier) else est
                def unwrap_params(params, est):
                    return {k[len('estimator__'):]: v for k,v in params.items() } if isinstance(est,CorruptTrainLabelsClassifier) else params

                # Put cv_models output into data frame
                results = results.append([
                    dict({
                        'Inner Split': int(''.join([s for s in cvr_k if s.isdigit()])),
                        'Estimator': unwrap_corrupt(cv_model.estimator),
                        'Estimator Label': config['label'], 
                        'Params': unwrap_params(params, cv_model.estimator),
                        'Params Index': params_index,
                        'Score': sc,
                        # No clean score
                    }, **split_dict)
                    for cv_model, config in zip(cv_models, estimator_configs)
                    for cvr_k,scores in cv_model.cv_results_.items()
                    if 'split' in cvr_k and 'test_score' in cvr_k
                    for params_index, (sc, params) in enumerate(zip(scores,cv_model.cv_results_['params']))
                ], ignore_index=True)

                # Get clean score for selected models by just copying from (Oracle) CV with Clean Train
                #  except for model '(Oracle) CV with Clean Train & Test' which should just get it's
                #  original best score, which is already clean
                cv_clean_scores = [
                    m.best_score_
                    if methodology[fi]['name'] == '(Oracle) CV with Clean Train & Test'
                    else clean_cv_models[di][si][mi].cv_results_['mean_test_score'][m.best_index_]
                    for mi, m in enumerate(cv_models)
                ]

                # Add best score and best clean score (i.e. clean score for selected model)
                results = results.append([
                    dict({
                        # No split inner
                        'Estimator': unwrap_corrupt(cv_model.estimator),
                        'Estimator Label': config['label'],
                        'Params': unwrap_params(cv_model.best_params_, cv_model.estimator),
                        'Score': cv_model.best_score_,
                        'Clean Score': clean_score,
                    },**split_dict)
                    for cv_model, clean_score, config in zip(cv_models, cv_clean_scores, estimator_configs)
                ], ignore_index=True)
                
                
                # Meta Estimator: Refit final estimator if necessary and score
                if meta_model_selection:
                    def unwrap_and_fit(est):
                        # Must refit underyling model without training label corruption
                        args = data_params_func(D,train)
                        args.pop('y_clean',None) # Pop off y_clean if exist
                        if isinstance(est, CorruptTrainLabelsClassifier):
                            return est.estimator.fit(**args)
                        return est.fit(**args)
                    meta_test_score = np.array([
                        unwrap_and_fit(m.estimator.set_params(**m.best_params_)).score(**data_params_func(D, test, True)) for m in cv_models
                    ])

                    # Meta Estimator: Score on clean data (unwrapping best estimator again if needed)
                    clean_data_params_func = [m['data_params_func'] for m in methodology if m['name'] == '(Oracle) CV with Clean Test'][0]
                    meta_test_score_clean = np.array([
                        unwrap_and_fit(m.estimator.set_params(**m.best_params_)).score(**clean_data_params_func(D,test, True)) 
                        for m in cv_models
                    ])
                else:
                    meta_test_score = np.array([0 for m in cv_models])
                    meta_test_score_clean = np.array([0 for m in cv_models])

                # Meta Estimator: Add outer CV results to data frame
                results = results.append([
                    dict({
                        # No split inner
                        # No estimator
                        'Estimator Label': 'MetaEstimator', # Meta estimator
                        'Params': {'model': 'CV(%s)' % config['label']},
                        'Params Index': params_index,
                        'Score': sc,
                        'Clean Score': sc_clean,
                    },**split_dict)
                    for params_index, (cv_model, config, sc, sc_clean) in enumerate(zip(cv_models, estimator_configs, meta_test_score, meta_test_score_clean))
                ], ignore_index=True)
                T_s.split('Outer split %d (%d inner splits)' % (si, cv_models[0].n_splits_ ))

            T_d.split('Synthetic corruption dataset %d' % di)

            # Meta Estimator: Fit meta estimator based on best model scores 
            #TODO Might want to split off as something as separate function 
            #  (so that results can be combined afterwards)
            # Coerce to numeric so that we can compute meta_mean_test_score_clean
            results['Clean Score'] = pd.to_numeric(results['Clean Score'])

            # Filter to MetaEstimator outer CV splits for each model
            sel = (
                (results['Methodology'] == methodology[fi]['name']) 
                & (results['Synthetic Corruption Index'] == di)
                & (results['Inner Split'].isnull())
                & (results['Outer Split'].notnull())
                & (results['Estimator Label'] == 'MetaEstimator')
            )

            # Group by Params Index (i.e. model index because meta estimator)
            meta_score_df = results.loc[sel,['Score','Params Index']].groupby('Params Index')
            meta_score_df_clean = results.loc[sel,['Clean Score','Params Index']].groupby('Params Index')
            meta_mean_test_score = meta_score_df.mean()
            meta_mean_test_score_clean = meta_score_df_clean.mean()

            # Select best model
            meta_best_index = meta_mean_test_score.idxmax(axis='index')[0]

            # Add meta estimator result to data frame
            results = results.append([
                dict({
                    # No split outer
                    # No split inner
                    # No estimator
                    'Estimator Label': 'MetaEstimator', # Meta estimator
                    'Params': {'model': 'CV(%s)' % estimator_configs[meta_best_index]['label']},
                    'Score': meta_mean_test_score.loc[meta_best_index,'Score'],
                    'Clean Score': meta_mean_test_score_clean.loc[meta_best_index,'Clean Score'],
                },**data_dict)
            ], ignore_index=True)

        T_m.split('Methodology %s' % methodology[fi]['name'])

    # Add param label
    results['Params Label'] = results['Params'].apply(
        lambda x: ','.join(['%s=%s' % (k,str(v)) for k,v in x.items() ])
    )
    T_top.split('Full synthetic experiment')
    return (results, D_arr)

def get_scores(R):
    def get_score_dict(methodology, estimator_label):
        base_sel = (R['Methodology'] == methodology) & (R['Estimator Label'] == estimator_label)

        # Get data for CV results/line plot
        if estimator_label == 'MetaEstimator':
            sel_cv = base_sel & (R['Inner Split'].isnull()) & (R['Outer Split'].notnull())
            sel_clean = base_sel & (R['Inner Split'].isnull()) & (R['Outer Split'].isnull())
        else:
            sel_cv = base_sel & (R['Inner Split'].notnull())
            sel_clean = base_sel & (R['Inner Split'].isnull()) & (R['Outer Split'].notnull())
        cv_data = (R.sort_values('Params Index') # Parentheses just for line continuation
                   .loc[sel_cv,['Params Label','Score']]
                   .groupby('Params Label',sort=False)
                    )
                  
        # Get data for best result/bar chart
        best_data = R.loc[sel_clean,'Clean Score']

        # Return results as dict
        cv_mean, cv_std = (cv_data.mean(), cv_data.std())
        return {'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_n': cv_data.count().iloc[0,0],
                'best_clean_mean': best_data.mean(),
                'best_clean_std': best_data.std(),
                'best_clean_n': best_data.count(),
                }
    
    # Construct nested dictionary
    sorted_methodology = R.sort_values('Methodology Index').groupby('Methodology',sort=False).count().index
    scores = {
        estimator_label: {
            methodology: get_score_dict(methodology,estimator_label)
            for methodology in sorted_methodology
        }
        for estimator_label in R['Estimator Label'].unique()
    }
    return scores

def get_err(std, n, confidence_level=0.95):
    # Note if DF is a vector, return scalar
    # if DF is matrix, return vector
    # DF is grouped so need to count fst
    return np.abs(stats.t.ppf((1-confidence_level)/2,n)
                  *std
                  /np.sqrt(n))

def plot_results(results, plot_cv=True, plot_bar=True):
    # Get scores
    scores = get_scores(results)
    
    # Loop through and plot results
    for estimator_label, estimator_scores in scores.items():
        # Plot CV results
        if plot_cv:
            # Handle sorting of values (only for estimator)
            if estimator_label == 'MetaEstimator':
                # Sort based on oracle score
                cv_mean_clean = np.array(estimator_scores['(Oracle) CV with Clean Test']['cv_mean']).transpose()[0]
                sorted_ind = np.argsort(cv_mean_clean)
            else:
                # Keep normal order
                n_scores = len(estimator_scores['(Oracle) CV with Clean Test']['cv_mean'].index)
                sorted_ind = range(0,n_scores)

            # **** HACKY ****
            # Handle the case of only one hyper parameter
            dict_as_string = estimator_scores[list(estimator_scores.keys())[0]]['cv_mean'].index[0]
            try:
                hyper_dict = {k: float(v_string) for kv in dict_as_string.split(',') for k, v_string in [kv.split('=',1)] }
            except ValueError:
                hyper_dict = {}
            n_hyper = len(hyper_dict)
            if n_hyper == 1:
                hyperparameter = next(iter(hyper_dict))
            else:
                hyperparameter = ''
            
            def methodology_rename(methodology,abbr=False):
                if abbr:
                    renames = {
                        'CV':'Standard',
                        'Corrupt CV (r=20%,sz=2.0+/-1.0%)':'Corrupt',
                        'Corrupt CV (r=20%,sz=0.0+/-0.0%)':'Corrupt',
                        'Corrupt CV':'Corrupt',
                        '(Oracle) CV with Clean Train & Test': 'Clean T&T',
                        '(Oracle) CV with Clean Test': 'Clean Test',
                    }
                else:
                    renames = {
                        'CV':'Standard',
                        'Corrupt CV (r=20%,sz=2.0+/-1.0%)':'Corrupt',
                        'Corrupt CV (r=20%,sz=0.0+/-0.0%)':'Corrupt',
                        'Corrupt CV':'Corrupt Validation',
                        '(Oracle) CV with Clean Train & Test': 'Clean Train & Test (Oracle)',
                        '(Oracle) CV with Clean Test': 'Clean Test (Oracle)',
                    }
                return renames[methodology]

            def estimator_rename(label):
                return label.replace(' (n_grams=1)','')
            
            estimator_label = estimator_rename(estimator_label)

            def get_x(sc_dict):
                return (['%.2g' % float(sc_dict['cv_mean'].index[i].split('=')[1]) for i in sorted_ind]
                        if n_hyper == 1 
                        else [sc_dict['cv_mean'].index[i] for i in sorted_ind])
            def get_y(sc_dict):
                return [1-sc_dict['cv_mean'].iloc[i,0] for i in sorted_ind]


            data = [
                go.Scatter(
                    name=methodology_rename(methodology), 
                    legendgroup=float('Oracle' in methodology),
                    x=get_x(sc_dict),
                    y=get_y(sc_dict),
                    mode=mode,
                    line=dict(dash=dash),
                    marker=dict(symbol=symbol),
                    error_y=dict(
                        type='data',
                        array=[np.array(get_err(sc_dict['cv_std'], sc_dict['cv_n']))[i] for i in sorted_ind],
                        visible=False, #(sc_dict['cv_n'] > 1), Just too much to squeeze into small plot
                        #opacity=0.5,
                    ) 
                )
                for (methodology, sc_dict), mode, dash, symbol in zip(
                        estimator_scores.items(),
                        ['lines+markers','lines+markers','lines+markers','lines+markers'],
                        ['solid','solid','dot','dot'],
                        ['circle','diamond','star','x'],
                    )
            ]
            layout = go.Layout(
                title='Hyperparameter Selection for %s' % estimator_label,
                #autosize=False,
                #width=500*0.8,
                #height=350*0.8,
                margin=dict(
                    l=50,
                    t=100,
                    r=10,
                    b=80,
                ),
                #paper_bgcolor='#7f7f7f',
                #plot_bgcolor='#c7c7c7',
                xaxis=dict(
                    title='Hyperparameter %s' % (hyperparameter) if n_hyper == 1 else 'Hyperparameters',
                    type='category',
                ), 
                yaxis=dict(
                    title='Validation Error',
                    tickformat='%',
                ),
                #showlegend=False,
                legend=dict(
                    orientation='h',
                    x=1,
                    y=1,
                    xanchor='right',
                    yanchor='bottom',
                    traceorder='grouped',
                ),
                shapes=[
                    dict(
                        type='line',
                        xref='x',
                        yref='paper',
                        x0=np.argmin(get_y(sc_dict)) + (-0.1 if 'Oracle' in methodology else 0.1),
                        x1=np.argmin(get_y(sc_dict)) + (-0.1 if 'Oracle' in methodology else 0.1),
                        y0=0,
                        y1=1,
                        line=dict(
                            dash=dash,
                            color=color,
                        ),
                    )
                    for (methodology, sc_dict), color, dash, symbol in zip(
                            estimator_scores.items(),
                            ['rgb(31,120,180)','rgb(255,127,0)','rgb(51,160,44)','rgb(227,26,28)'  ],
                            ['solid','solid','dot','dot'],
                            ['circle','diamond','star','x'],
                    )
                ]
            )
            py.iplot(go.Figure(data=data,layout=layout))

        # Plot best results
        if plot_bar:
            bars = [
                go.Bar(
                    x=[methodology_rename(methodology, True)],
                    y=[1-sc_dict['best_clean_mean']],
                    error_y=dict(
                        type='data', 
                        array=[get_err(sc_dict['best_clean_std'], sc_dict['best_clean_n'])],
                        visible=bool(sc_dict['best_clean_n'] > 1),
                    )
                )
                for methodology, sc_dict in estimator_scores.items()
                if '& Test' not in methodology
            ]
            layout = go.Layout(
                title='True Error',
                autosize=False,
                width=120,
                height=300,
                margin=dict(
                    l=50,
                    t=30,
                    r=10,
                    b=80,
                ),
                #paper_bgcolor='#7f7f7f',
                #plot_bgcolor='#c7c7c7',
                showlegend=False,
                yaxis=dict(
                    title='Held-Out Error on Clean Labels',
                    tickformat='%'
                ),
            )
            #if len(bars) > 4:
                #layout.height = 800
            py.iplot(go.Figure(data=bars, layout=layout))
