import sys
import pickle
import os
sys.path.append("../../tools/")
import numpy as np
from collections import defaultdict
from sklearn import preprocessing
from feature_format import featureFormat, targetFeatureSplit
from sklearn.cross_validation import train_test_split
import sklearn.pipeline
from sklearn.cross_validation import StratifiedShuffleSplit
import sklearn.grid_search
from sklearn.linear_model import LogisticRegression
import math
from sklearn.decomposition import PCA

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

finance_features = ['deferral_payments' , 'expenses' , 'deferred_income' , 'restricted_stock_deferred','loan_advances' ,'other' , 'director_fees', 'bonus' ,'restricted_stock' ,'total_stock_value' ,'long_term_incentive','salary','total_payments','exercised_stock_options']

def test_classifier(clf, dataset, feature_list, folds = 1000):
    data = featureFormat(dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    cv = StratifiedShuffleSplit(labels, folds, random_state = 42)
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv: 
        features_train = []
        features_test  = []
        labels_train   = []
        labels_test    = []
        for ii in train_idx:
            features_train.append( features[ii] )
            labels_train.append( labels[ii] )
        for jj in test_idx:
            features_test.append( features[jj] )
            labels_test.append( labels[jj] )
        
        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print "Warning: Found a predicted label not == 0 or 1."
                print "All predictions should take value 0 or 1."
                print "Evaluating performance for processed predictions:"
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0*(true_positives + true_negatives)/total_predictions
        precision = 1.0*true_positives/(true_positives+false_positives)
        recall = 1.0*true_positives/(true_positives+false_negatives)
        f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
        f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    except:
        print "Got a divide by zero when trying out:", clf
        print "Precision or recall may be undefined due to a lack of true positive predicitons."

os.chdir('../../final_submission/')

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)    
my_dataset = data_dict


del my_dataset['TOTAL']
del my_dataset['THE TRAVEL AGENCY IN THE PARK']
del my_dataset['LOCKHART EUGENE E']

# def add_new_features(x):

# 	if(x['from_messages']!='NaN' and x['from_this_person_to_poi']!='NaN'):
# 		ratio_from_x_to_poi = x['from_this_person_to_poi'] / float(x['from_messages'])
# 	else:
# 		ratio_from_x_to_poi = 0.

# 	if(x['to_messages']!='NaN' or x['from_poi_to_this_person']!='NaN'):
# 		ratio_from_poi_to_x = x['from_poi_to_this_person'] / float(x['to_messages'])
# 	else:
# 		ratio_from_poi_to_x = 0.		

# 	if(x['to_messages']!='NaN' and x['shared_receipt_with_poi']!='NaN'):
# 		ratio_shared_receipt_with_poi = x['shared_receipt_with_poi'] / float(x['to_messages'])
# 	else:
# 		ratio_shared_receipt_with_poi = 0.

# 	return ratio_from_x_to_poi , ratio_from_poi_to_x , ratio_shared_receipt_with_poi

# for i in my_dataset:
# 	my_dataset[i]['ratio_to_poi'] = 0.
# 	my_dataset[i]['ratio_from_poi'] = 0.
# 	my_dataset[i]['ratio_shared_receipt'] = 0.

# for i in my_dataset : 
# 	ratio_to_poi , ratio_from_poi , ratio_shared_receipt = add_new_features(my_dataset[i])
# 	my_dataset[i]['ratio_to_poi'] = ratio_to_poi
# 	my_dataset[i]['ratio_from_poi'] = ratio_from_poi
# 	my_dataset[i]['ratio_shared_receipt'] = ratio_shared_receipt


# for i in my_dataset:
# 	print('\n')
# 	for features in my_dataset[i] :
# 		print(features , my_dataset[i][features])
# 	print('\n\n')
# 	break




# def log_features(x) :

#     d = defaultdict(lambda : 0.)
#     d.clear()
#     for features in x:
#         if(features in finance_features) : 
#             if(x[features] != 'NaN'):
#                 if(x[features]!=0) :
#                     d['log_' + str(features)] = math.log(abs(x[features]) , 10)
#             else:
#                 d['log_' + str(features)] = 0
#     return d

# for i in my_dataset : 
#     d = log_features(my_dataset[i])
#     for features in d:
#         my_dataset[i][features] = d[features]

def normalized_features(x):

    d = defaultdict(lambda : 0.)

    for features in x:
        if(features in finance_features) : 
            if(x[features]!='NaN' and x['salary']!='NaN' and features!='salary'):
                d['normalized_' + str(features)] = (x[features] / float(x['salary']))
            else:
                d['normalized_' + str(features)] = 0.
    return d


# for i in my_dataset :
#     for features in my_dataset[i].keys():
#         if(features in finance_features and features!='salary'):
#             my_dataset[i]['normalized_' + str(features)] = 0.

for i in my_dataset:
    d = normalized_features(my_dataset[i])
    for features in my_dataset[i].keys():
        if(features in finance_features and features!='salary'):
            my_dataset[i]['normalized_' + str(features)] = d['normalized_' + str(features)]

features_list = ['poi']

my_dataset = data_dict
features_all = set()

for x in my_dataset : 
    for features in my_dataset[x] :
            features_all.add(features)
try :
    features_all.remove('poi')
except KeyError:
    pass


features_list = ['poi']

for i in features_all : 
    if(i!='email_address') :
        features_list.append(i)
   

print('\n Total feature size : ' , len(features_list))


# features_list.remove('ratio_from_poi')
# features_list.remove('ratio_shared_receipt')

data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest

select = SelectKBest()

from sklearn.ensemble import ExtraTreesClassifier

clf = LogisticRegression()

scaler = preprocessing.MinMaxScaler()

pca = PCA()

steps = [('feature_selection' , select) ,('scaler' , scaler), ('PCA' , pca) ,  ('classifier' , clf)]

pipeline = sklearn.pipeline.Pipeline(steps)

parameters = parameters = dict(feature_selection__k= [30,31,32,'all'] , 
                               classifier__penalty = ['l2'] ,
                               classifier__max_iter = [100,200,500,1000 , 10000 , 100000] ,
                               classifier__solver = ['liblinear'] ,
                               classifier__fit_intercept = [True , False] ,
                               classifier__dual = [True],
                               PCA__n_components =  [30] ,
                               PCA__random_state = [42] ,
                               PCA__whiten = [True , False])

sss = StratifiedShuffleSplit(labels, 5, test_size=0.3, random_state=60)

cv = sklearn.grid_search.GridSearchCV(pipeline, param_grid = parameters , scoring = 'f1' ,cv = sss)

features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

cv.fit(features, labels)

X_new = cv.best_estimator_.named_steps['feature_selection']

# Get SelectKBest scores, rounded to 2 decimal places, name them "feature_scores"
feature_scores = ['%.2f' % elem for elem in X_new.scores_ ]
# Get SelectKBest pvalues, rounded to 3 decimal places, name them "feature_scores_pvalues"
feature_scores_pvalues = ['%.3f' % elem for elem in  X_new.pvalues_ ]
# Get SelectKBest feature names, whose indices are stored in 'X_new.get_support',
# create a tuple of feature names, scores and pvalues, name it "features_selected_tuple"
features_selected_tuple=[(features_list[i+1], feature_scores[i], feature_scores_pvalues[i]) for i in X_new.get_support(indices=True)]

# Sort the tuple by score, in reverse order
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)

# Print
print ' '
print 'Not Scaled - Selected Features, Scores, P-Values'
for i in features_selected_tuple :
	print(i)



#y = cv.predict(features_test)

#print(str('\n Best estimators : \n')+ str(cv.best_estimator_))

clf  = cv.best_estimator_

print('\nBest features : \n' + str(clf) + '\n\n')

test_classifier(clf , my_dataset , features_list)



#Liblinear - with dual
# Pipeline(steps=[('feature_selection', SelectKBest(k=14, score_func=<function f_c
# lassif at 0x00000000167B2BA8>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=Tru
# e, fit_intercept=False,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.86760       Precision: 0.51129      Recall: 0.15850 F1: 0.24
# 198     F2: 0.18387
#         Total predictions: 15000        True positives:  317    False positives:
#   303   False negatives: 1683   True negatives: 12697

#liblinear - without dual
# Pipeline(steps=[('feature_selection', SelectKBest(k=15, score_func=<function f_c
# lassif at 0x00000000167B2BA8>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=Fal
# se, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l1', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.86253       Precision: 0.43725      Recall: 0.10800 F1: 0.17
# 322     F2: 0.12715
#         Total predictions: 15000        True positives:  216    False positives:
#   278   False negatives: 1784   True negatives: 12722

#newton - cg 

# Pipeline(steps=[('feature_selection', SelectKBest(k=14, score_func=<function f_c
# lassif at 0x0000000016702BA8>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('classifier', LogisticRegression(C=1.0, class_weight=None, dual=Fal
# se, fit_intercept=False,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='newton-cg', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.86760       Precision: 0.51129      Recall: 0.15850 F1: 0.24
# 198     F2: 0.18387
#         Total predictions: 15000        True positives:  317    False positives:
#   303   False negatives: 1683   True negatives: 12697

# Pipeline(steps=[('feature_selection', SelectKBest(k=4, score_func=<function f_cl
# assif at 0x0000000016683BA8>)), ('scaler', MinMaxScaler(copy=True, feature_range
# =(0, 1))), ('classifier', ExtraTreesClassifier(bootstrap=False, class_weight=Non
# e, criterion='gini',
#            max_depth=None, max_features='auto', ...stimators=5, n_jobs=1, oob_sc
# ore=False, random_state=None,
#            verbose=0, warm_start=False))])
#         Accuracy: 0.85820       Precision: 0.43682      Recall: 0.21950 F1: 0.29
# 218     F2: 0.24375
#         Total predictions: 15000        True positives:  439    False positives:
#   566   False negatives: 1561   True negatives: 12434

#--------------------------------------------------------------------------------------

#POI 
#ExtraRandom

# Pipeline(steps=[('feature_selection', SelectKBest(k=3, score_func=<function f_cl
# assif at 0x0000000016705BA8>)), ('scaler', MinMaxScaler(copy=True, feature_range
# =(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=2, random_
# state=42,
#   svd_solver='auto', tol=0.0, whiten=False)), ('classifier'...timators=10, n_job
# s=1, oob_score=False, random_state=None,
#            verbose=0, warm_start=False))])
#         Accuracy: 0.86533       Precision: 0.48936      Recall: 0.23000 F1: 0.31
# 293     F2: 0.25727
#         Total predictions: 15000        True positives:  460    False positives:
#   480   False negatives: 1540   True negatives: 12520

#------------------------------------------------------------------------------------
#Liblinear - with dual
# Pipeline(steps=[('feature_selection', SelectKBest(k=27, score_func=<function f_c
# lassif at 0x0000000016787BA8>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=4, random
# _state=42,
#   svd_solver='auto', tol=0.0, whiten=False)), ('classifier...ty='l2', random_sta
# te=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.69520       Precision: 0.28122      Recall: 0.82650 F1: 0.41
# 965     F2: 0.59555
#         Total predictions: 15000        True positives: 1653    False positives:
#  4225   False negatives:  347   True negatives: 8775
#------------------------------------------------------------------------------------

#liblinear - dual
# Pipeline(steps=[('feature_selection', SelectKBest(k=31, score_func=<function f_c
# lassif at 0x0000000016820BA8>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=6, random
# _state=42,
#   svd_solver='auto', tol=0.0, whiten=False)), ('classifier...ty='l2', random_sta
# te=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.69373       Precision: 0.27935      Recall: 0.82100 F1: 0.41
# 686     F2: 0.59158
#         Total predictions: 15000        True positives: 1642    False positives:
#  4236   False negatives:  358   True negatives: 8764

#Newton

# Pipeline(steps=[('feature_selection', SelectKBest(k=27, score_func=<function f_c
# lassif at 0x00000000167FFBA8>)), ('scaler', MinMaxScaler(copy=True, feature_rang
# e=(0, 1))), ('PCA', PCA(copy=True, iterated_power='auto', n_components=4, random
# _state=42,
#   svd_solver='auto', tol=0.0, whiten=False)), ('classifier...ty='l2', random_sta
# te=None, solver='newton-cg', tol=0.0001,
#           verbose=0, warm_start=False))])
#         Accuracy: 0.69520       Precision: 0.28122      Recall: 0.82650 F1: 0.41
# 965     F2: 0.59555
#         Total predictions: 15000        True positives: 1653    False positives:
#  4225   False negatives:  347   True negatives: 8775