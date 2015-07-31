'''

Library for process data_dict for sk-learn machine learning

'''

def scale_features(features):
    """
    Arguments:
        Scale features using the MinMax algorithm
    Retures:
        scaled features in numpy.ndarray
    """
    # scale features via min-max
    from sklearn import preprocessing
    scaler = preprocessing.MinMaxScaler()
    features = scaler.fit_transform(features)

    return features


def prep_features(df, features_list):

    """
    Arguments:
        load dataframe (or dictionary), and features_list
    return
        scaled features, labels in numpy.ndarray, and
        scaled features, labels in pandas dataframe
    """
    from feature_format import featureFormat, targetFeatureSplit
    import pandas as pd
    # for pandas dataframe
    df1 = df[features_list]
    features_df = df1.drop('poi', axis=1)#.astype(float)  # new features (pandas dataframe)
    labels_df = df1['poi']  # new labels (pandas dataframe)
    features_df_scaled = scale_features(features_df) # scale features
    # for dictionary
    df2 = df[features_list]
    data_dict_new = df2.T.to_dict()  # data_dict (final)
    features_dic = features_df.copy()
    X_features = list(features_dic.columns)
    features_list_new = ['poi'] + X_features  # selected features list (final)
    data = featureFormat(data_dict_new, features_list_new, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    features = scale_features(features)

    return features, labels, features_df_scaled, labels_df
