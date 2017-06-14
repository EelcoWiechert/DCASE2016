

import os
import time
import pickle
from multiprocessing import Process
from DataLoader import print_time_stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
import numpy as np
from confusion_matrix import *
import Output
from Model import *
from SegmentHandler import *
from collections import Counter
import xgboost
from FinalEval import *

N_WORKERS                   = 4         # Number of processes to perform prediction in. (Select 1 to disable multiprocessing.)

# return most common element in a list
def most_common(lst):
    # return Counter(lst.flat).most_common(1)[0][0]
    return [max(set(lst), key=lst.count)]

def ensemble_method(segment_predictions):
    return most_common(segment_predictions)

def ensemble_predictions(y_pred):
    # Create list containing one list of 1500 predicted classes for each segment
    y_pred_folded = data_fold(y_pred)

    # This is our output
    segment_y_pred = []

    # For each list of 1500 per segment, ensemble these predictions to the final class
    # - using method defined above!
    for segment_predictions in y_pred_folded:
        listed = list(map(lambda x: x[0], segment_predictions))
        segment_y_pred.append(ensemble_method(listed))

    return segment_y_pred


# Function for prediction in a separate process
# Does the predictions, then dumps them so that they can be loaded by the parent process again
def prediction_worker(clf, data, dump_filename, proc_num):
    predictions = []
    count = 0.0
    for item in data:
        count += 1

        # Print progress
        if count % 1024 == 0:
            print("Process " + str(proc_num) + ": " + str(round(count / len(data), 4) * 100) + " % done")

        predictions.append(clf.predict([item]))

    # Dump the results
    pickle.dump(predictions, open(dump_filename, "wb"), protocol=2)


def predict(classifier, x_test):
    # Making predictions
    # ------------------
    print('Making predictions...\n')
    start_time = time.time()

    x_test = np.array(x_test)
    y_predict = []

    # Do the prediction
    if N_WORKERS == 1:
        # Predict in single process / single thread
        count = 0
        for case in x_test:
            count += 1
            if count % 500 == 0:
                print(count)
            y_predict.append(classifier.predict([case]))

    else:
        # Use multiple processes for prediction
        bucket_size = round(len(x_test) / N_WORKERS + 1, 0)
        start = 0
        jobs = []
        dump_filenames = []
        for i in range(0, N_WORKERS):
            stop = int(min(float(len(x_test)), start + bucket_size))
            dump_filename = "tmp_prediction_dump_" + str(i)
            dump_filenames.append(dump_filename)
            process = Process(target=prediction_worker, args=(classifier, x_test[start:stop], dump_filename, i))
            jobs.append(process)
            start = stop

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        # Load the finished predictions
        for dump_filename in dump_filenames:
            y_predict += pickle.load(open(dump_filename, "rb"))
            os.remove(dump_filename)

    print_time_stats("Making predictions", start_time)
    return (y_predict, str((time.time() - start_time)))

def fit_model(classifier, x_train, y_train):
    # Fitting the model
    # -----------------
    print('Fitting model...')
    start_time = time.time()

    classifier.fit(x_train, y_train)

    print_time_stats("Fitting model", start_time)
    return (classifier, str((time.time() - start_time)))
