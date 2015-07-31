
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
