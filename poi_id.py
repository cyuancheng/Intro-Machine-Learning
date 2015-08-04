#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("./tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data
import pandas as pd
import numpy as np

from feature_format import featureFormat
from feature_format import targetFeatureSplit

from poi_model import *
from poi_dataprocess import *

# sklearn packages below
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest

from sklearn.cross_validation import StratifiedShuffleSplit

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments',
       'exercised_stock_options', 'bonus', 'restricted_stock',
       'shared_receipt_with_poi', 'restricted_stock_deferred',
       'total_stock_value', 'expenses', 'loan_advances', 'from_messages',
       'other', 'from_this_person_to_poi', 'poi', 'director_fees',
       'deferred_income', 'long_term_incentive',
       'from_poi_to_this_person'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
df = pd.DataFrame.from_dict(data_dict, orient='index') # convert to pandas dataframe
df = df.replace('NaN', np.nan)  # replace with np.nan

### Task 2: Remove outliers
df1 = df.copy() # make a new dataframe, df1
df1 = df1.drop('TOTAL', 0) # remove 'TOTAL'
df1 = df1.drop('THE TRAVEL AGENCY IN THE PARK', 0)
df1 = df1.drop('LOCKHART EUGENE E', 0)
df1 = df1.drop('email_address', 1) # remove email address entry
df1 = df1.apply(lambda x: x.fillna(x.median()), axis=0) # replace nan with median value

### Task 3: Create new feature(s)
df2 = df1.copy() # make a new dataframe, df2

df2['fraction_from_poi'] = df2['from_poi_to_this_person'] /(df2['from_messages'] + df2['to_messages'] )
df2['fraction_to_poi'] = df2['from_this_person_to_poi'] /(df2['from_messages'] + df2['to_messages'] )

df2['fraction_emails_with_poi'] = (df2['from_poi_to_this_person']+ df2['from_this_person_to_poi']) / \
                                (df2['from_messages'] + df2['to_messages'] )
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing

# Feature selection using SelectKBest

best_feature = get_k_best(df2, features_list, 6) # SelectKBest
df_best_feature = pd.DataFrame.from_dict(best_feature, orient='index') #convert to pandas dataframe
selected_features =  list(df_best_feature.sort(0, ascending=False).index.values)[0:6]
print "Select top " + str(5) + " features"
print selected_features


# prepare the data in dictionary
df3 = df2[selected_features]
data_dict_new = df3.T.to_dict()

### Task 4: Try a varity of classifiers

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# preliminary test for selected classifiers
from poi_dataprocess import *
features, labels, features_df_scaled, labels_df = \
prep_features(df2, selected_features, True)

cv = StratifiedShuffleSplit(labels_df, n_iter=100, test_size=0.1, random_state = 42)
# preliminary test for selected classifiers

models = [DecisionTreeClassifier(), LinearSVC(), AdaBoostClassifier(),
    RandomForestClassifier(), KNeighborsClassifier(), LogisticRegression()]

#from poi_model import *
for clf in models:
    accuracy = evaluate_model(clf, features_df_scaled, labels_df, 'accuracy' , cv)
    precision = evaluate_model(clf, features_df_scaled, labels_df, 'precision' , cv)
    recall = evaluate_model(clf, features_df_scaled, labels_df, 'precision' , cv)

    print "---------------------------------------------------"
    print "classifier: {0},  score = {1}, precision = {2}, recall = {3}".\
    format(clf, accuracy, precision, recall)

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
print "#####  Start parameter optimzation of Adaboost using GridSearchCV ###############################"
########################################
# uncommment below for find the optimal parameters of Adaboost using GridSearchCV
features, labels, features_df_scaled, labels_df = \
prep_features(df2, selected_features, False)
cv = StratifiedShuffleSplit(labels_df, n_iter=100, test_size=0.1, random_state = 42)

pipeline, params = clf_adaboost()

GridSearch = GridSearchCV(pipeline, param_grid=params, n_jobs = 1, cv=cv ,
 scoring='precision', verbose=0)
GridSearch.fit(features_df_scaled, labels_df)
print "#####  best_estimator for Adaboost  #####"
clf_best_adaboost = GridSearch.best_estimator_
print clf_best_adaboost
print "------------------------------------------------------------------------"
print "Cross-validated precision = ",GridSearch.best_score_
print "------------------------------------------------------------------------"
print "Best Parameters = ", GridSearch.best_params_
print "------------------------------------------------------------------------"
print "Scores using optimal parameters:"
accuracy = evaluate_model(clf_best_adaboost, features_df_scaled, labels_df, 'accuracy' , cv)
precision = evaluate_model(clf_best_adaboost, features_df_scaled, labels_df, 'precision' , cv)
recall = evaluate_model(clf_best_adaboost, features_df_scaled, labels_df, 'recall' , cv)
print 'Accuracy ={0},  Precision = {1}, Recall = {2}'.format(accuracy, precision, recall)


print "#####  Start valudation using test_classifier() ###############################"
test_classifier(clf_best_adaboost, data_dict_new, selected_features)

### Dump your classifier, dataset, and features_list so
### anyone can run/check your results.

dump_classifier_and_data(clf_best_adaboost, data_dict_new, selected_features)
