
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GMM
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from DataLoader import *
from ModelEvaluator import *
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier


# Things to do:
# -------------
# 1) Check section "File paths:" in files DataLoader.py and Output.py
#    (that the paths are correct for you and the directories exist)
#    - if you want to reload the data from files, set it at RELOAD_FROM_FILES in DataLoader.py
# 2) Select how much data you want to run the classifier on at SAMPLE_SIZE below
# 3) See the lines lower marked with TODO
#    - select the algorithm and describe the parameters there
#    - or use for loop to try various parameters, but do not forget to adjust the scenario_label each time!
# 4) Run it
# 5) Enjoy

# Start script
# ------------
print("\nStarted script: " + str(time.ctime()) + "\n")

# Load data like this
# - they will be loaded and dumped
# - the dumps will be then used in the evaluate_folds method
fold_names = load_folds()

# TODO: Set name of the classifier
scenario_name = "Ensemble"

# TODO: Describe the parameters
# (this will appear in the summary file and confusion_matrix filename!)
scenario_label = "default_params"
parameter_name = "Voting_hard"
parameter_value = 'dt & gnb'

# TODO: Select classifier
clf1 = DecisionTreeClassifier(max_leaf_nodes=12000, min_samples_leaf=25)
clf2 = GaussianNB() #RandomForestClassifier(max_depth=20, min_samples_split=15)
clf3 = KNeighborsClassifier(n_neighbors=10)


clf = VotingClassifier(estimators=[('dt', clf1), ('gnb', clf2)], voting='hard')

# Run the evaluation
evaluate_folds(clf, fold_names,
               scenario_name, scenario_label,
               parameter_name, parameter_value)


# End script
# ------------
print("\nFinished script: " + str(time.ctime()) + "\n")

# Configuration
# -------------
#SAMPLE_SIZE                 = 1       # Choose how much data to work with (1 = all data)

# Start script
# ------------
#print("\nStarted script: " + str(time.ctime()) + "\n")

# Load data like this
# - they will be loaded and dumped
# - the dumps will be then used in the evaluate_folds method
#fold_names = load_folds()

# Load data like this
#x, y = load_data(standardize=False, normalize=False,
#                 sample_size=SAMPLE_SIZE,
#                 pca=False, components=0,
#                 features_limit=2)



# Basic setup as suggested by scikit-learn, but I believe that this code only runs the first classifier (Decision Trees),
# and gives that as output for the ensemble

# TODO: Set name of the classifier
# scenario_name = "VotingCombo"

# TODO: Describe the parameters
# (this will appear in the summary file and confusion_matrix filename!)
# scenario_label = "DT+RF+KNN"
# parameter_name = "AsInStage2"
# parameter_value = 0

# TODO: Select classifier
# clf1 = DecisionTreeClassifier(max_leaf_nodes=12000, min_samples_leaf=25)
# clf2 = RandomForestClassifier(max_depth=20, min_samples_split=15)
# clf3 = KNeighborsClassifier(n_neighbors=10)

# eclf = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('knn', clf3)], voting='hard')

# evaluate_model(clf3, x, y,
#                scenario_name, scenario_label,
#                parameter_name, parameter_value,
#                SAMPLE_SIZE, scale="")

# evaluate_model(eclf, x, y,
#                scenario_name, scenario_label,
#                parameter_name, parameter_value,
#                SAMPLE_SIZE, scale="")



# Another code snippet, more transparent, but still has low accuracy scores for the basic classifiers, and therefor
# also the ensemble.

#clf1 = DecisionTreeClassifier(max_leaf_nodes=12000, min_samples_leaf=25)
#clf2 = RandomForestClassifier(max_depth=20, min_samples_split=15)
#clf3 = KNeighborsClassifier(n_neighbors=10)



#eclf = VotingClassifier(estimators=[('dt', clf1), ('rf', clf2), ('knn', clf3)], voting='hard')

#for clf, label in zip([clf1, clf2, clf3, eclf],['DT', 'RF', 'KNN','Ensemble']):
    #scores = cross_val_score(clf, x, y, cv=2, scoring='accuracy')
    #print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

# End script
# ------------
#print("\nFinished script: " + str(time.ctime()) + "\n")
