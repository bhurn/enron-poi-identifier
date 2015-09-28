#!/usr/bin/python

import sys
import pickle
sys.path.append("/home/bhurn/Data_Analyst_P4/ud120-projects/tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# List of all features for univariate selection
all_features = ['poi', 'to_messages', 'deferral_payments', 'expenses',
                 'deferred_income', 'long_term_incentive',
                 'restricted_stock_deferred', 'shared_receipt_with_poi',
                 'loan_advances', 'from_messages', 'other', 'director_fees',
                 'bonus', 'total_stock_value',
                 'from_poi_to_this_person', 'from_this_person_to_poi',
                 'restricted_stock', 'salary', 'total_payments',
                 'exercised_stock_options', 'f_t_ratio', 't_s_ratio']

# poi and three selected features for final classifier
features_list = ['poi', 'exercised_stock_options', 'total_stock_value',
                'bonus']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
# Code to view summary stats
'''
from scipy.stats import describe
for feat in all_features:
    feat_summary=[]
    for name in data_dict:
        if data_dict[name][feat] != "NaN":   
            feat_summary.append(data_dict[name][feat])
    print feat
    print describe(feat_summary)
'''

data_dict.pop("TOTAL",0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK",0)

# Recheck summary stats after removing outliers
'''
from scipy.stats import describe
for feat in all_features:
    feat_summary=[]
    for name in data_dict:
        if data_dict[name][feat] != "NaN":   
            feat_summary.append(data_dict[name][feat])
    print feat
    print describe(feat_summary)
'''

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

# Add ratios of messages from poi to received and to poi to sent
for name in data_dict:
    from_poi = data_dict[name]['from_poi_to_this_person']
    to_poi = data_dict[name]['from_this_person_to_poi']
    to_mess = data_dict[name]['to_messages']
    sent_mess = data_dict[name]['from_messages']
    if from_poi != "NaN" and to_mess != "NaN" and to_mess > 0:
        data_dict[name]['f_t_ratio'] = float(from_poi) / float(to_mess)
    else:
        data_dict[name]['f_t_ratio'] = 0 
    if to_poi != "NaN" and sent_mess != "NaN" and sent_mess > 0:
        data_dict[name]['t_s_ratio'] = float(to_poi) / float(sent_mess)
    else:
        data_dict[name]['t_s_ratio'] = 0 

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

'''
# Implement univariate feature selection and print scores
# Use all_features as features_list to evaluate all 21 features
from sklearn.feature_selection import SelectKBest 
feats = SelectKBest(k='all')
feats.fit(features, labels)
print feats.scores_
'''

### Tasks 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()    # Provided to give you a starting point. Try a varity of classifiers.
'''
# Try decision tree
from sklearn import tree    
clf = tree.DecisionTreeClassifier(min_samples_split=6)

# Try SVM
from sklearn.svm import SVC
clf = SVC(kernel="linear")
'''

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)