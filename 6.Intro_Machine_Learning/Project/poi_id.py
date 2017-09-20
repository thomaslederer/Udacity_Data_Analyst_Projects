
# coding: utf-8

# In[1]:

#!/usr/bin/python

# Load packages and scripts
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


# In[2]:

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
    
#Exploring the dataset:

#There are 146 people in the dictionary
print ("Number of employees: " + str(len (data_dict)))

#There are 18 POIs
poi = []
for k, v in data_dict.items():
    if v["poi"] == True:
        poi.append(k)
print ("Number of POIs: " + str(len (poi)))

# There are 21 total features:
possible_features = set()
for v in data_dict.values():
    for k in v:
        if k != "poi":
            possible_features.add(k)
print ("Number of possible features: " + str(len(possible_features)))


# In[3]:

### Task 1: Select features.
    
#Add all existing features to features list
features_list = ['poi','salary'] 

for v in data_dict.values():
    for k in v:
        if k not in features_list:
            features_list.append(str(k))

#Remove 'email_address' - this string is a unique indentifier for each employee
features_list.remove('email_address')




# In[4]:

### Task 2: Remove outliers 

#Remove aggregate datapoint:
del data_dict['TOTAL']

#Remove non-employee datapoint:
del data_dict ["THE TRAVEL AGENCY IN THE PARK"]



# In[5]:

### More outlier removal


# Change NaNs to zeros. Although the featureformat function function would otherwise 
# perform this function, doing it prior makes it easier to  parse data for further outliers


for k,v in data_dict.items():
    for sub_k, sub_v in v.items():
         if sub_v == "NaN": 
            v [sub_k] = 0   


# Check for any cases where financial information does not add up correctly:

total_payments_sum = [
"salary",
"deferral_payments",
"bonus",
"expenses",
"loan_advances",
"other",
"director_fees",
"deferred_income",
"long_term_incentive"]

total_stock_value_sum = [
"exercised_stock_options",
"restricted_stock",
"restricted_stock_deferred"]


total_payments_problems = []
total_stock_problems = []

for key, value in data_dict.items():
    test = 0
    for sub_k in value:
        if sub_k in total_payments_sum:
            test += value[sub_k]
    if test == value["total_payments"]:
        continue
    else:
        total_payments_problems.append(key)
                

for key, value in data_dict.items():
    test = 0
    for sub_k in value:
        if sub_k in total_stock_value_sum:
            test += value[sub_k]
    if test == value["total_stock_value"]:
        continue
    else:
        total_stock_problems.append(key)

            
print ("Stock Problems: " + str(total_stock_problems))
print ("Payments Problems: " + str(total_payments_problems))



#Since lists happen to be the same, can use either to remove further outliers:
for employee in total_stock_problems:
    del data_dict[employee]



# In[6]:

### Task 3: Create new feature(s)

#Ratio values created for each subtotal
for k,v in data_dict.items():
    if v["total_payments"] != 0 :
        v["percent_salary"] = float(float(v["salary"])/ float(v["total_payments"]))
        v["percent_deferral_payments"] = float(float(v["deferral_payments"])/ float(v["total_payments"]))
        v["percent_bonus"] = float(float(v["bonus"])/ float(v["total_payments"]))
        v["percent_expenses"] = float(float(v["expenses"])/ float(v["total_payments"]))
        v["percent_loan_advances"] = float(float(v["loan_advances"])/ float(v["total_payments"]))
        v["percent_other"] = float(float(v["other"])/ float(v["total_payments"]))
        v["percent_director_fees"] = float(float(v["director_fees"])/ float(v["total_payments"]))
        v["percent_deferred_income"] = float(float(v["deferred_income"])/ float(v["total_payments"]))
        v["percent_long_term_incentive"] = float(float(v["long_term_incentive"])/ float(v["total_payments"]))
        
    else:
        v["percent_salary"] = 0
        v["percent_deferral_payments"] = 0
        v["percent_bonus"] = 0
        v["percent_expenses"] = 0
        v["percent_loan_advances"] = 0 
        v["percent_other"] = 0
        v["percent_director_fees"] = 0
        v["percent_deferred_income"] = 0
        v["percent_long_term_incentive"] = 0
        

for k,v in data_dict.items():
    if v["total_stock_value"] != 0:
        v["percent_exercised_stock_options"] = float(float(v["exercised_stock_options"])/ float(v["total_stock_value"]))
        v["percent_restricted_stock"] = float(float(v["restricted_stock"])/ float(v["total_stock_value"]))
        v["percent_restricted_stock_deferred"] = float(float(v["restricted_stock_deferred"])/ float(v["total_stock_value"]))
    else:
        v["percent_exercised_stock_options"] = 0
        v["percent_restricted_stock"] = 0
        v["percent_restricted_stock_deferred"] = 0

# Re-run code to add new feature to feature_list:

for v in data_dict.values():
    for k in v:
        if k not in features_list:
            features_list.append(str(k))
features_list.remove('email_address')


# In[7]:

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


# In[8]:

### Task 4: Create classifier

# Import necessary packages and create objects for pipeline.

#We will test two classifiers: Decision Tree and K Nearest Neighbor, and use - respectively - select K best
#and PCA to reduce the number of features.


###'scaling'###
from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler()


### "princomp" ###
from sklearn.decomposition import PCA
pca = PCA ()


####"classifier_nn":###
from sklearn.neighbors import KNeighborsClassifier
nn = KNeighborsClassifier()


####"kbest"###
from sklearn.feature_selection import SelectKBest
#Varied independently selector with top 6 features produces best results
selector = SelectKBest()


####"classifier_dt":###
from sklearn import tree
dt = tree.DecisionTreeClassifier(random_state = 123)





# In[9]:

#Initial evaluation using standard split of data, 30% reserved for testing

from sklearn.cross_validation import train_test_split

    
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score

def split_test (clf, print_results = False):
    features_train, features_test, labels_train, labels_test =     train_test_split(features, labels, test_size=0.3, random_state=42)
    clf = clf.fit(features_train, labels_train)
    pred = clf.predict(features_test)
    accuracy = round(accuracy_score (labels_test, pred), 3)
    recall = round (recall_score (labels_test, pred), 3)
    precision = round (precision_score (labels_test, pred), 3)
    f1 = round (f1_score(labels_test, pred), 3)
    if print_results == True:
        counter = 0
        for i in pred:
            if i == 1.:
                counter +=1

        counter0 = 0
        for i in labels_test:
            if i == 1.:
                counter0 +=1
        print clf
        print ("total number of POIs predicted: " + str(counter) +"/" + str(len(pred)))
        print ("real number of POIs " + str(counter0) +"/" + str(len(labels_test)))
        print ("acc.: " + str(accuracy))
        print ("recall: " + str(recall))
        print ("prec. :" + str(precision))
        print ("f1 :" + str(f1))
    else:
        return (accuracy, recall, precision, f1)


# In[69]:

# Create Pipeline for Decision Tree classifier and find optimal number of features/k value in selectKbest.

from sklearn.pipeline import Pipeline

results_list = []
n = 10
best_f1_score = 0
optimal_k = 0

while n >= 1:
    results = {}
    results[n] = {"accuracy":0, "recall":0, "precision":0,"f1":0}
    selector = SelectKBest(k = n)
    clf_dt =  Pipeline(steps=[('kbest',selector), ("classifier_dt", dt)])
    accuracy, recall, precision, f1 = split_test (clf_dt)
    results[n]["accuracy"] = accuracy
    results[n]["recall"] = recall
    results[n]["precision"] = precision
    results[n]["f1"] = f1
    results_list.append (results)
    if f1 > best_f1_score:
        best_f1_score = f1
        optimal_k = n
    n = n -1

    
#Overview of results with number of features from 1 - 10
print ("Overview of results:")

for i in results_list:
    print i

    
    
# With k set at 6, we get very good intial results from this algoritm:
print "optimal k: " + str(optimal_k) 

selector = SelectKBest(k = optimal_k)
clf_dt =  Pipeline(steps=[('kbest',selector), ("classifier_dt", dt)])


split_test (clf_dt, print_results = True)


# In[185]:

# Scores for 6 features selected by decision tree classifier : 

print sorted(selector.scores_, reverse = True) [0:6]

print sorted(dt.feature_importances_, reverse = True) 


# In[70]:

# Repeat process for k nearest neighbor and pca. The algorithm performs best with 2 components; 
#although it does not score as well as the decision tree classifier, it performs very well using 
#the split method of validation. 

results_list = []
n = 10
best_f1_score = 0
optimal_n_components = 0

while n >= 1:
    results = {}
    results[n] = {"accuracy":0, "recall":0, "precision":0,"f1":0}
    pca = PCA (n_components = n)
    clf_nn =  Pipeline(steps=[('scaling',scalar),('princomp',pca), ("classifier_nn", nn)])
    accuracy, recall, precision, f1 = split_test (clf_nn)
    results[n]["accuracy"] = accuracy
    results[n]["recall"] = recall
    results[n]["precision"] = precision
    results[n]["f1"] = f1
    results_list.append (results)
    if f1 > best_f1_score:
        best_f1_score = f1
        optimal_n_components = n
    n = n -1

    
#Overview of results with number of features from 1 - 10
print ("Overview of results:") 

for i in results_list:
    print i
    
    
    
print "optimal n components : " + str(optimal_n_components) 

pca = PCA (n_components = optimal_n_components)
clf_nn =  Pipeline(steps=[('scaling',scalar),('princomp',pca), ("classifier_nn", nn)])

split_test (clf_nn, print_results = True)



# In[187]:

# Rankings for variances in PCA

print (pca.explained_variance_ratio_)


# In[43]:

# However, further evaluating each algorithm using a cross method of validation, 
#only the k nearest neighbor passes a .3 test for precison and recall


PERF_FORMAT_STRING = "\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\tRecall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\tFalse negatives: {:4d}\tTrue negatives: {:4d}"


def test_classifier (clf, folds = 1000, print_results = True):
    '''Note: this function modified from tester.py included in course materials'''
    from sklearn.cross_validation import StratifiedShuffleSplit
    data = featureFormat(my_dataset, features_list, sort_keys = True)
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
           
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    if print_results == True:
        print clf
        print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
        print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
        print ""
    else:
        return (total_predictions,accuracy,precision,recall,f1,f2)
   
test_classifier (clf_nn)
test_classifier (clf_dt)


# In[160]:

# Task 5: Tune classifier for best parameters:

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
sss = StratifiedShuffleSplit(100, random_state=42)


params_nn = {'classifier_nn__n_neighbors': [1,3,5,7,9]}
params_dt = {"kbest__k": [1,2,3,4,5,6,7,8],
            "classifier_dt__min_samples_split":[2,3,4]}

def tune_classifier (clf, params, score_function):
    '''Note: this function adopted from forum page on Udacity
    https://discussions.udacity.com/t/stratifiedshufflesplit-and-train-test-split/255763 '''
    gs = GridSearchCV(clf, params, cv = sss, scoring = score_function)
    for train_index, test_index in sss.split(features, labels):
        features_train       = []
        features_test        = []
        labels_train        = []
        labels_test         = []
        for index in train_index:
            features_train.append(features[index])
            labels_train.append(labels[index])
        for index in test_index:
            features_test.append(features[index])
            labels_test.append(labels[index])

        gs.fit(features_train, labels_train)

    clf = gs.best_estimator_
    return clf



# In[91]:

# Configuring the decision tree algorithm in Gridsearch (under Stratified Shuffle Split methods of vaidation)
# returns a SelectKBest value of 5 and min_samples_split of three.
tuned_dt = tune_classifier(clf_dt, params_dt, "f1")


# In[190]:

# These configurations improve the algorithm just enough to achieve a .3 score on precision and recall using the more 
# rigorous Stratified Shuffle Split for scoring.

test_classifier (tuned_dt)


# In[162]:

# The default configuration for KNearestNeighbor of n_neighbors = 5 happens to also be 
# the optimal setting for the algorithm

tuned_nn  = tune_classifier(clf_nn, params_nn, None)


# In[163]:

# Thus the previous scores are not improved 
test_classifier (tuned_nn)


# In[68]:

# Since varying n_components for a pca operation within gridsearch takes a very long time to 
# compute, code below tests the number of components from 1 to 10. The code is the same as before
# except the method of validation is different.

results_list = []
n = 10
best_f1_score = 0
optimal_n_components = 0

while n >= 1:
    results = {}
    results[n] = {"accuracy":0, "recall":0, "precision":0,"f1":0, "f2":0}
    pca = PCA (n_components = n)
    clf_nn =  Pipeline(steps=[('scaling',scalar),('princomp',pca), ("classifier_nn", nn)])
    total_predictions,accuracy,precision,recall,f1,f2 = test_classifier(clf_nn, print_results    = False)
    results[n]["accuracy"] = round(accuracy,3)
    results[n]["recall"] = round(recall, 3)
    results[n]["precision"] = round(precision,3)
    results[n]["f1"] = round(f1,3)
    results[n]["f2"] = round(f2,3)
    results_list.append (results)
    if f1 > best_f1_score:
        best_f1_score = f1
        optimal_n_components = n
    n = n -1
    

    
#Overview of results with number of features from 1 - 10
print ("Overview of results:") 

for i in results_list:
    print i
    
# Here again, 2 components for pca produces the best results for our classifer and outperforms
#the decision tree algorithm.
print "optimal n components : " + str(optimal_n_components) 

pca = PCA (n_components = optimal_n_components)
clf_nn =  Pipeline(steps=[('scaling',scalar),('princomp',pca), ("classifier_nn", nn)])

test_classifier (clf_nn, print_results = True)
    


# In[192]:

### Task 6: Dump classifier, dataset, and features_list to reproduce results.


clf = clf_nn
dump_classifier_and_data(clf, my_dataset, features_list)
    


# In[1]:

# As a final excercise, we can test the importance of the created ratio features to the final algorithm 
# by re-runing the same test without these features.

original_features = []

#Remove all features created in step 3 
for feature in features_list:
    feature = feature.strip()
    if feature.find("percent"):
        original_features.append(feature)
    
    
    
features_list = original_features
test_classifier (clf_nn, print_results = True)


# In[ ]:



