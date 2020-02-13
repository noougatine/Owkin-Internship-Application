#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def k_cross_val(pena, selection = False, features = None) :
    """
    5 fold cross validation.
    """
    cphs = []
    scores = []
    x1, x2, y1, y2 = get_data_set(x, y, 1/5, selection, features)
    x2, x3, y2, y3 = get_data_set(x2, y2, 1/4, selection, features)
    x3, x4, y3, y4 = get_data_set(x3, y3, 1/3, selection, features)
    x4, x5, y4, y5 = get_data_set(x4, y4, 1/2, selection, features)
    
    xs, ys = [x1, x2, x3, x4, x5], [y1, y2, y3, y4, y5]
    
    for k in range(5) :
        x_train = pd.DataFrame(columns = x1.columns)
        y_train = pd.DataFrame(columns = y1.columns)
        for i in range(5) :
            if (i!=k) :
                x_train = pd.concat([x_train, xs[i]], axis = 0)
                y_train = pd.concat([y_train, ys[i]], axis = 0)
        cph = CoxPHFitter(penalizer = pena).fit(pd.concat([x_train, y_train], axis = 1),
                                                duration_col = 'SurvivalTime', event_col='Event')
        y_pred = cph_pred(cph.predict_expectation(xs[k]))

        if not(np.all(y_pred.iloc[:,0].values)) : # for some reasons sometimes predicted lifetime is null
            ys[k] = y_test.drop(y_pred.iloc[np.where(y_pred.iloc[:,0].values==0)[0]].index, 0)
            y_pred = y_pred.drop(y_pred.iloc[np.where(y_pred.iloc[:,0].values==0)[0]].index, 0)
        
        cphs.append(cph)
        scores.append(cindex(ys[k], y_pred))
        
    return (cphs, scores)

def best_model_cross_val(n_trial, penas, selection = False, features = None) :
    """
    Return "best" trained model for a given penalization and a given set of features.
    """
    best_score = 0
    for pena in penas :
        for i in range(n_trial) :
            cphs, scores = k_cross_val(pena, selection = True, features = features)
            cph, score = cphs[np.argmax(scores)], scores[np.argmax(scores)]
            if (score > best_score) :
                best_cph = cph
                best_score = score
                best_pena = pena
    return(best_cph, best_score, best_pena)

