# In[1]

import numpy as np 
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import train_test_split
import tensorflow as tf
from metrics_t9gbvr2 import cindex

tf.__version__


# In[2]:  Loading the data

# Radiomics
radiomics_train = pd.read_csv('radiomics_x_train.csv', index_col = 0)
radiomics_train = radiomics_train.iloc[2:,:]
radiomics_train.index = radiomics_train.index.map(int)

# Clinical data
clinical_data_train = pd.read_csv('clinical_data_x_train.csv', index_col = 0)
clinical_data_train = clinical_data_train[['Tstage', 'Mstage', 'Nstage', 'age']]
clinical_data_train.index = clinical_data_train.index.map(int)

# Survival time
y = pd.read_csv('output_VSVxRFU_y_train.csv', index_col = 0)
y.index = y.index.map(int)

# Train data
x = pd.concat([radiomics_train, clinical_data_train], axis = 1)
x = x.fillna(x.mean()) # Fill Na values (age)

def get_data_set(x, y, train_size=0.75, selection=False, features=None) :
    """
    Parameters
    ----------
    x : pd.DataFrame
        DataFrame imported from the CSV for the radiomics
    y : pd.DataFrame
        DataFrame imported from the CSV for the survival times + events.
    train_size : 0<float<1, optional
        Set the size of the training set. The default is 0.75.
    selection : bool, optional
        Feature selection enabling. The default is False.
    features : pd.Index, optional
        The features to select. The default is None.

    Returns
    -------
    x_train, x_test, y_train, y_test : pd.DataFrame

    """
    x_train, x_test = train_test_split(x, train_size = train_size)
    y_train, y_test = y.loc[x_train.index], y.loc[x_test.index]
    if selection :
        x_train = pd.DataFrame(x_train[features])
        x_test = x_test[features]
    else :
        x_train = pd.DataFrame(x_train)
    return(x_train, x_test, y_train, y_test)

def cph_pred(y_pred) :
    """
    Create a DataFrame that can be used as input for the score function from a "blank" DataFrame.

    Parameters
    ----------
    y_pred : pd.DataFrame
        "Raw" DataFrame.

    Returns
    -------
    y_pred : pd.DataFrame
    """
    y_pred.columns = ['SurvivalTime']
    y_pred.index.names = ['PatientID']
    y_pred['Event'] = 0
    return(y_pred) 

def cross_val(pena, train_size = 0.75, selection = False, features = None) :
    """
    Hold out method.
    ----------
    pena : float>0
        Penalization coefficient for the L2 penalization.
    train_size : 0<float<1, optional
        Set the size of the training set. The default is 0.75.
    selection : bool, optional
        Feature selection enabling. The default is False.
    features : pd.Index, optional
        The features to select. The default is None.

    Returns
    -------
    cph : COXPHFitter.
        The Cox model. lifeline object
    score : float
        Score from the metrics.

    """
    x_train, x_test, y_train, y_test = get_data_set(x, y, train_size, selection, features)
    cph = CoxPHFitter(penalizer = pena).fit(pd.concat([x_train, y_train], axis = 1),
                                            duration_col = 'SurvivalTime', event_col='Event')
    y_pred = cph_pred(cph.predict_expectation(x_test))
    
    if not(np.all(y_pred.iloc[:,0].values)) : # for some reasons sometimes predicted lifetime is null
        y_test = y_test.drop(y_pred.iloc[np.where(y_pred.iloc[:,0].values==0)[0]].index, 0)
        y_pred = y_pred.drop(y_pred.iloc[np.where(y_pred.iloc[:,0].values==0)[0]].index, 0)
        
    return (cph, cindex(y_test, y_pred))
    

# In[3]: First score


n_pena = 10
n_trial = 10
best_score = 0

for pena in np.logspace(-2,3,n_pena) :
    score = 0
    for i in range(n_trial) :
        _, s = cross_val(pena)
        score += s/n_trial
    if score > best_score :
        best_score = score
        best_pena = pena

print(best_score, best_pena)

"""
Submission of the prediction with this model gave a poor score (0.67).
"""
    

# In[4]: Feature selection with Pearson correlation coefficient.

def pearson_selection(n_test, selection_proba) :
    """
    Return the selected features with Pearson correlation over n_test trial.

    Parameters
    ----------
    n_test : int
        Number of dataset to evaluate on.
    selection_proba : 0<float<1
        Determines if a features should be selected or not.

    Returns
    -------
    selected_features : pd.Index
        The selected features.
    """
    sum_columns = np.zeros(57)
    for i in range(n_test) :     
        x_train, x_test, y_train, y_test = get_data_set(x, y)
        x_train = x_train.iloc[:,:57] 
        x_train_float = x_train.astype('float32')

        corr = np.corrcoef(x_train_float, rowvar = False)

        columns = np.full((corr.shape[0],), True, dtype=bool)
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[0]):
                if corr[i,j] >= 0.9:
                    if columns[j]:
                        columns[j] = False

        columns = np.multiply(columns, 1)
        sum_columns = columns + sum_columns

    selected_features = x_train.columns[(sum_columns > selection_proba * n_test)]
    
    return selected_features


# In[5]: 
    """
    Selecting the best features, then the best model by lopping over penalization
    coefficients and cross validation.
    """

n_pena = 10
n_test_crossval = 10
n_test_corr = 1000
selection_proba = 0.9

selected_features = pearson_selection(n_test_corr, selection_proba)

print(selected_features.shape[0], 'features out of', x_train.shape[1], 'were selected.')

best_score = 0
for pena in np.logspace(-2,3,n_pena) :
    score = 0
    for i in range(n_test_crossval) :
        _, s = cross_val(pena, selection = True, features = selected_features)
        score += s/n_trial
    if score > best_score :
        best_score = score
        best_pena = pena

print('With selected features, best score of', best_score, 'with pena of', best_pena)


# In[6]:

def best_model(n_trial, penas, train_size = 0.75, selection = False, features = None) :
    """
    Return "best" trained model for a given penalization and a given set of features.

    Parameters
    ----------
    n_test : int
        Number of dataset to evaluate on.
    penas : list of float>0
        The penlisation coefficients to test.
    train_size : TYPE, optional
    train_size : 0<float<1, optional
        Set the size of the training set. The default is 0.75.
    selection : bool, optional
        Feature selection enabling. The default is False.
    features : pd.Index, optional
        The features to select. The default is None.

    Returns
    -------
    best_cph : COXPHFitter.
        The Cox model. lifeline object
    best_score : float
        Score from the metrics with the best model.
    best_pena : float>0
        The penalisation coefficient which gave the best score
    """
    best_score = 0
    for pena in penas :
        for i in range(n_trial) :
            cph, score = cross_val(pena, selection = True, features = selected_features)
            if (score > best_score) :
                best_cph = cph
                best_score = score
                best_pena = pena
    return(best_cph, best_score, best_pena)


# In[7]:

    """
    Same as before, but we test more different parameters
    """
    
penas = [50, 100, 500]
n_test_crossval = 500

selection_correl = [0.8, 0.85, 0.9]
n_test_corr = 100
 
best_score = 0

for proba in selection_correl :
    selected_features = pearson_selection(n_test_corr, proba)
    cph, score, pena = best_model(n_test_crossval, penas, selection = True, features = selected_features)
    if score > best_score :
        best_cph = cph
        best_score = score
        best_pena = pena
        best_selected_features = selected_features 
        best_selection_corr = proba
print('Best score of', best_score, 'with', best_selected_features.shape[0], 'features.')


# In[8]:

"""
Submission
"""


# Test for the public dataset

# Radiomics
radiomics_test = pd.read_csv('radiomics_x_test.csv', index_col = 0)
radiomics_test = radiomics_test.iloc[2:,:]
radiomics_test.index = radiomics_test.index.map(int)

# Clinical data
clinical_data_test = pd.read_csv('clinical_data_x_test.csv', index_col = 0)

clinical_data_test = clinical_data_test[['Tstage', 'Mstage', 'Nstage', 'age']]
clinical_data_test.index = clinical_data_test.index.map(int)

# Features selection
x_test = pd.concat([radiomics_test, clinical_data_test], axis = 1)

x_test = x_test[best_selected_features]
x_test = x_test.fillna(x_test.mean()) # Fill Na values (age)

# Test the model
lifetime_pred = best_cph.predict_expectation(x_test)
lifetime_pred.columns = ['SurvivalTime']
lifetime_pred.index.names = ['PatientID']
lifetime_pred.index = x_test.index
lifetime_pred['Event'] = None

# Prediction
lifetime_pred.to_csv('y_test_selected_features.csv')


"""
Score of 0.7198
"""

# In[9]:
"""
I decided to analyze a bit the model with the function from lifeline.
"""

cph.print_summary()

"""
We can see that some features are "useless" (coefficient = 0). We can try to find 
them and remove them.
"""

# In[10]:

"""
Loop over the best_model function to select the features that are usually 
useless in the best model. Quite computational expensive because it is really badly
written (many loops).
"""

removing_indices = np.zeros(best_selected_features.shape[0])
n_trial = 10
for i in range(n_trial) :
    cph, _, _ = best_model(n_trial, [best_pena], train_size = 0.75, selection = True, features = best_selected_features)
    indices = np.where(np.abs(cph.params_) < 0.01)
    for j in range(len(indices)) :
        removing_indices[indices[j]] += 1
x_train, _, _, _ = get_data_set(x, y, selection=True, features=best_selected_features)
removed_features = x_train.columns[removing_indices > n_trial * 0.95]
print(removed_features)


# In[11]:

"""
Finding the best model with the selected + removed features so far.
"""

n_trial = 100
penas = np.linspace(best_pena/2, best_pena*3/2, 10) #np.logspace(-1,2,10)
cph, score, pena = best_model(n_trial, penas, selection = True, features = best_selected_features.drop(removed_features))
print(score)


# In[12]:

"""
Submission
"""

# Radiomics
radiomics_test = pd.read_csv('radiomics_x_test.csv', index_col = 0)
radiomics_test = radiomics_test.iloc[2:,:]
radiomics_test.index = radiomics_test.index.map(int)

# Clinical data
clinical_data_test = pd.read_csv('clinical_data_x_test.csv', index_col = 0)

clinical_data_test = clinical_data_test[['Tstage', 'Mstage', 'Nstage', 'age']]
clinical_data_test.index = clinical_data_test.index.map(int)

# Features selection
x_test = pd.concat([radiomics_test, clinical_data_test], axis = 1)

x_test = x_test[best_selected_features]
x_test = x_test.fillna(x_test.mean()) # Fill Na values (age)

# Test the model
lifetime_pred = cph.predict_expectation(x_test)
lifetime_pred.columns = ['SurvivalTime']
lifetime_pred.index.names = ['PatientID']
lifetime_pred.index = x_test.index
lifetime_pred['Event'] = None

# Prediction
lifetime_pred.to_csv('y_test_selected_features_final.csv')

"""
Score is 0.728
"""


