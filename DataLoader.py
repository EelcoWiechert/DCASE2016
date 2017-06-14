
import os
import arff
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split
from sklearn import decomposition
from sklearn import preprocessing
from SegmentHandler import *
from Model import *

# -------------
# Configuration
# -------------

RELOAD_FROM_FILES           = False    # If yes, data will be loaded from source files. If false, data will be loaded from serialized files to speed up the process.

# File paths:
PATH_FILES                  = '../data/'                            # Where the weka files are
PATH_DUMP_DATA              = '../serialized_data/data'             # Where the serialized data should be saved / loaded from
PATH_DUMP_LABELS            = '../serialized_data/labels'           # Where the serialized labels should be saved / loaded from
PATH_EVAL_SETUP             = '../evaluation_setup/'                # Where the eval setup definitions are
PATH_DUMP_FOLDS             = '../serialized_data/fold'             # Where the folds are dumped
PATH_DUMP_FOLDS_NAMES       = '../serialized_data/fold_names.p'     # Where the folds are dumped


# ----------------
# Functions
# ----------------

# Calculate elapsed time
def print_time_stats(action, past_time):
    now = time.time()
    passed = now - past_time
    print("")
    print("Done:       " + action)
    print("Time taken: " + str(round(passed, 2)) + " seconds")
    print("----")
    print("")



def load_filenames(source):
    with open(source, "r") as fp:
        filenames = []
        for line in fp:
            name = line.split("\t")[0].strip()
            name = name.split("/")[1].split(".")[0] + ".arff"
            filenames += [name]
    return filenames

def load_contents(files, dir = PATH_FILES):
    x = []
    y = []
    count = 0
    for file_name in files:
        # print progress
        count += 1
        if count % 100 == 0:
            print('%s of %s' % (count, len(files)))

        for mfcc in arff.load(open(dir + file_name, mode="r"))["data"]:  # for every mfcc in a file
            x.append(mfcc[:-1])  # data
            y.append(mfcc[-1])   # labels

    return (x, y)

def load_folds():
    if RELOAD_FROM_FILES == False:
        loaded_folds = pickle.load(open(PATH_DUMP_FOLDS_NAMES, "rb"))
        return loaded_folds

    loaded_folds: [Fold] = []
    for i in range(1,5):
        print("\n\nLoading: Fold " + str(i) + "\n===")
        start_time = time.time()

        # Names of the setup files
        source_train    = PATH_EVAL_SETUP + "fold" + str(i) + "_train.txt"
        source_test     = PATH_EVAL_SETUP + "fold" + str(i) + "_test.txt"
        source_eval     = PATH_EVAL_SETUP + "fold" + str(i) + "_evaluate.txt"

        # Grab filenames for this fold
        fn_train    = load_filenames(source_train)
        fn_test     = load_filenames(source_test)
        fn_eval     = load_filenames(source_eval)

        print_time_stats("Loaded filenames: Fold " + str(i), start_time)
        start_time = time.time()


        # Load data for this fold

        fold = Fold()
        fold.train_x, fold.train_y = load_contents(fn_train)
        print_time_stats("Loaded training set: Fold " + str(i), start_time)
        start_time = time.time()

        fold.test_x,  fold.test_y  = load_contents(fn_test)
        print_time_stats("Loaded test set: Fold " + str(i), start_time)
        start_time = time.time()

        fold.eval_x,  fold.eval_y  = load_contents(fn_eval)
        print_time_stats("Loaded evaluation set: Fold " + str(i), start_time)
        start_time = time.time()

        dump_name =  PATH_DUMP_FOLDS + str(i) + ".p"
        pickle.dump(fold, open(dump_name, "wb"), protocol=2)
        print_time_stats("Dumped data: Fold " + str(i), start_time)

        loaded_folds += [dump_name]

    # Dump and return names of the loaded folds
    pickle.dump(loaded_folds, open(PATH_DUMP_FOLDS_NAMES, "wb"), protocol=2)
    return loaded_folds


if __name__ == "__main__":
    load_folds()




def load_data(
        pca=False, components=0,
        standardize=False, normalize=False, sample_size = 1,
        features_limit = 60):

    # Load or deserialize files
    # -------------------------

    if RELOAD_FROM_FILES:
        # Load file names
        # ---------------
        print('Loading file names...')
        start_time = time.time()

        files = []
        for filename in os.listdir(PATH_FILES):
            files.append(filename)

        print('Loaded %s filenames.' % len(files))
        print_time_stats("Loading filenames", start_time)

        # Load data
        # ---------
        print('Loading data...')
        start_time = time.time()

        x = []
        y = []
        count = 0
        for file_name in files:
            # print progress
            count += 1
            if count % 100 == 0:
                print('%s of %s' % (count, len(files)))

            for mfcc in arff.load(open(PATH_FILES + file_name, mode="r"))["data"]:  # for every mfcc in a file
                x.append(mfcc[:-1])  # data
                y.append(mfcc[-1])  # labels

        print_time_stats("Loading data", start_time)

        # Serialize data
        # --------------
        print('Serializing data...')
        start_time = time.time()

        pickle.dump(x, open(PATH_DUMP_DATA, "wb"), protocol=2)
        pickle.dump(y, open(PATH_DUMP_LABELS, "wb"), protocol=2)

        print_time_stats("Serializing data", start_time)


    else:
        # Deserialize data
        # ----------------
        print('Deserializing data...')
        start_time = time.time()

        x = pickle.load(open(PATH_DUMP_DATA, "rb"))
        y = pickle.load(open(PATH_DUMP_LABELS, "rb"))

        print_time_stats("Deserializing data", start_time)


    # Cut the features if desired
    # ---------------------------
    if features_limit > 0 and features_limit < 60:
        x = list(map(lambda frame: frame[:features_limit], x))

    # Standardize the data
    # ----------------------
    if standardize:
        x = preprocessing.scale(x)

    # Normalize the data
    # ----------------------
    if normalize:
        x = preprocessing.normalize(x)


    # Sample the data
    # ----------------------

    # fold data to prevent cutting segments
    x_folded = data_fold(x)
    y_folded = labels_fold(y)


    rng = np.random.RandomState(31337)
    if sample_size != 1:
        unused_x, x_folded, unused_y, y_folded = train_test_split(x_folded, y_folded, test_size=sample_size, random_state=rng)

    # unfold data to previous format
    x = data_unfold(x_folded)
    y = labels_unfold(y_folded)

    # Perform principle component analysis to reduce the features
    # ----------------------
    if pca:
        print('Performing principe component analysis')

        pca = decomposition.PCA(n_components=components)
        pca.fit(x)
        x = pca.transform(x)
        print(x)

        print('Overall: %s %% of the variance is explained by these variables' % round(
            np.sum(pca.explained_variance_ratio_)) * 100, 2)
        print('Detail: %s' % pca.explained_variance_ratio_)


    return (x, y)

