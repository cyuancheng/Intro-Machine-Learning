## Identifying Fraud from Enron Email | Intro Machine Learning
- Author:  Chiyuan Cheng (cyuancheng AT gmail DOT com)
- Last updated: August 4, 2015

### Information

The goal of this project is to develop machine learning algorithm, together with scikit-learn Python module, to predict the person of interest (POI) of a fraud from the email and financial (E+F) dataset. POIs were ‘individuals who were indicted, reached a settlement, or plea deal with the government, or testified in exchange for prosecution immunity.’  The E+F dataset from the Enron Corpus was used as features for the POI prediction.

### Summary
After testing the performance of multiple algorithms, **AdaBoost** (with optimal Decision Tree estimator and PCA) was used as an optimized algorithm based on its hignest scores of precision and recall among other algorithms (> 0.3).

### Data Manipulation

I used pandas and Python to clean and explore the E+F dataset, including removing outliners and filling missing values with median values for each feature.

### Algorithm 
1. Select the following 5 features based on SelectKBest, PCA, and DecisionTreeClassifier: ```` 'poi', 'exercised_stock_options', 'total_stock_value', 'bonus', 'salary', 'deferred_income' ````

2. Cross-validation using StratifiedShuffleSplit and StratifiedKFold
3. Machine learning algorithm tuning based on the performance of the following classifiers:
 - Decision Tree
 - Adaboost
 - Random Forest
 - KNeighbors
 - LinearSVC
 - Logistic Regression

4. Fine tune the parameters of the two best performed classifiers (LinearSVC and Adaboost).


### Files

- **Project Summary** ([html](http://cyuancheng.github.io/Intro-Machine-Learning/))
- **Questions** ([html](http://htmlpreview.github.io/?https://github.com/cyuancheng/Intro-Machine-Learning/blob/master/P4_questions.html))
- **References** ([txt](reference.txt))  
- **Codes**
	- [poi_id.py](poi_id.py)
	- [poi_model.py](poi_model.py)
	- [poi_dataprocess.py](poi_dataprocess.py)
	- [Work flow ipynb file](http://nbviewer.ipython.org/github/cyuancheng/Intro-Machine-Learning/blob/master/Project4_ML_workflow.ipynb) (work flow file)
- **Certificate**  [(pdf)](certificate-3.pdf)
