
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


# Things to do:
# -------------
# 1) Check section "File paths:" in files DataLoader.py and Output.py
#    (that the paths are correct for you and the directories exist)
#    - if you want to reload the data from files, set it at RELOAD_FROM_FILES in DataLoader.py
# 2) See the lines lower marked with TODO
#    - select the algorithm and describe the parameters there
#    - or use for loop to try various parameters, but do not forget to adjust the scenario_label each time!
# 3) Run it
# 4) Enjoy


# Start script
# ------------
print("\nStarted script: " + str(time.ctime()) + "\n")

# Load data like this
# - they will be loaded and dumped
# - the dumps will be then used in the evaluate_folds method
fold_names = load_folds()

# TODO: Set name of the classifier
scenario_name = "NaiveBayes"

# TODO: Describe the parameters
# (this will appear in the summary file and confusion_matrix filename!)
scenario_label = "default_params"
parameter_name = "_"
parameter_value = 0

# TODO: Select classifier
clf = GaussianNB()

# Run the evaluation
evaluate_folds(clf, fold_names,
               scenario_name, scenario_label,
               parameter_name, parameter_value)

# End script
# ------------
print("\nFinished script: " + str(time.ctime()) + "\n")
