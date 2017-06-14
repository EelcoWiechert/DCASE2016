
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from DataLoader import *
from ModelEvaluator import *


# Here, scnarios are defined (do not configure scenario here, but lower, in Configuration)
SCENARIO_NAIVE_BAYES    = 1
SCENARIO_KNN            = 2
SCENARIO_RANDOM_FOREST  = 3
SCENARIO_ADABOOST       = 4
SCENARIO_DECISION_TREE  = 5
SCENARIO_MOST_FREQUENT  = 6
SCENARIO_RANDOM         = 7



# -------------
# Configuration
# -------------

# General
SCENARIO                    = 6         # Select scenario here
SAMPLE_SIZE                 = 1         # Choose how much data to work with (1 = all data)
STANDARDIZE_DATA            = False  # Set true to standardize the data, otherwise false
NORMALIZE_DATA              = False   # Set true to normalize the data, otherwise false

#Principle component analysis
PCA                         = True      # If true, component analysis will be performed
NUM_OF_COMPONENTS           = 2         # Number of components the vector is reduced to
# kNN
KNN_N_NEIGH                 = 3         # Select the number of neighbours for the kNN classifier

# name scale
if STANDARDIZE_DATA:
    scale = 'standardized'
elif NORMALIZE_DATA:
    scale = 'normalized'
else:
    scale = 'normal'

# ----------------
# Start the script
# ----------------
print("\nStarted script: " + str(time.ctime()) + "\n")

if STANDARDIZE_DATA==True and  NORMALIZE_DATA==True:
    print('ERROR!!!!!!! Data can not be standardized and normalized')

x, y = load_data(STANDARDIZE_DATA, NORMALIZE_DATA, SAMPLE_SIZE)


# Init selected scenario
if SCENARIO == 1:
    scenario_name = 'naive_bayes'
    clf = GaussianNB()
elif SCENARIO == 2:
    scenario_name = 'knn'
    clf = KNeighborsClassifier(n_neighbors = KNN_N_NEIGH)
elif SCENARIO == 3:
    scenario_name = 'random_forrest'
    clf = RandomForestClassifier()
elif SCENARIO == 4:
    scenario_name = 'adaboost'
    clf = AdaBoostClassifier()
elif SCENARIO == 5:
    scenario_name = 'decision_tree'
    clf = DecisionTreeClassifier()
elif SCENARIO == 6:
    scenario_name = 'most_frequent'
    clf = DummyClassifier(strategy='most_frequent')
elif SCENARIO == 7:
    scenario_name = 'random'
    clf = DummyClassifier(strategy='uniform')
else:
    print("\n!!!")
    print("Selected invalid scenario, defaulting to Naive Bayes.")
    print("!!!\n")
    scenario_name = 'naive_bayes'
    clf = GaussianNB()

evaluate_model(clf, x, y, scenario_name, 'default_params', SAMPLE_SIZE, scale)



