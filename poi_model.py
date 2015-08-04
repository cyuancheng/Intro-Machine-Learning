
"""
Library for returning sk-learn pipelines and parameters
for use in predictive model building.
This module provides pipeline and parameters creation functions in building
POI prediction models
"""

def get_k_best(df, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    #feature, label = feature_format_scale(data_dict, features_list)
    from poi_dataprocess import *
    from feature_format import featureFormat, targetFeatureSplit

    data_dict_new = df[features_list].T.to_dict()

    data = featureFormat(data_dict_new , features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    #df = df[features_list]
    #features = df.drop('poi', axis=1)#.astype(float)
    #labels = df['poi']

    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(features)

    from sklearn.feature_selection import SelectKBest
    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])

    return k_best_features

def clf_adaboost():
    '''
    AdaBoost
    return: pipeline, and optimal parameters
    '''
    from sklearn.pipeline import Pipeline
    from sklearn.decomposition import PCA
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.tree import DecisionTreeClassifier

    pipeline = Pipeline([
                            ('pca', PCA()),
                            ('clf', AdaBoostClassifier())
                       ])

    params = {
        'pca__n_components': [1, 2, 3,4,5, 'mle'],
        'clf__base_estimator' : [
            DecisionTreeClassifier(criterion='gini', max_depth=None,
            min_samples_leaf=1, min_samples_split=0.31622776601683794,
            random_state=42, splitter='random')
                                ],    # optimial estimator
        "clf__n_estimators": [1,2,3,4,5],
        "clf__learning_rate" :[0.5,1,1.5],#np.logspace(-1, 1, 8),
        "clf__random_state" : [42]

            }

    return pipeline, params



def evaluate_model(model, features, labels, score , cv):
    '''
        fucntion to evaluate the score of model
        return: score(can be accuracy, precision, or recall)
    '''
    from sklearn.cross_validation import StratifiedKFold, cross_val_score
    import numpy as np
    return np.mean(cross_val_score(
        model,
        features,
        labels,
        scoring= score,
        cv=cv,
        n_jobs=1))
