

## In this file, the given model is trained and cross validated.
#  Then, output is generated with the help of Output.py file.

# do not use sklearn.cross_validation




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
from Prediction import *

CROSS_VALIDATE  = False      # Whether to perform full cross validation
DO_PREDICTIONS  = True
DO_FINAL_EVAL   = False      # Whether to also predict classes for final eval data



def evaluate_folds(classifier, folds: [Fold],
                   algorithm_name, algorithm_label,
                   parameter_name, parameter_value,
                   sample_size=1, scale=1):

    # Here, information about evalutation will be stored
    model_info = ModelInfo()
    model_info.name = algorithm_name
    model_info.label = algorithm_label
    model_info.sample_size = sample_size
    model_info.param_name = parameter_name
    model_info.param_value = parameter_value

    print("")
    print("==============================================================")
    print("Evaluating model:  " + algorithm_name)
    print("With parameter:    " + parameter_name + " = " + str(parameter_value))
    print("==============================================================")

    cross_validation_round = 1

    eval_predictions = []

    for fold_name in folds:
        print("")
        print("Cross validation round " + str(cross_validation_round))
        print("========================")

        # Load the dumped data
        fold: Fold = pickle.load(open(fold_name, "rb"))

        # convert to numpy arrays
        fold.train_x    = np.array(fold.train_x)
        fold.train_y    = np.array(fold.train_y)
        fold.test_x     = np.array(fold.test_x)
        fold.test_y     = np.array(fold.test_y)
        fold.eval_x     = np.array(fold.eval_x)
        fold.eval_y     = np.array(fold.eval_y)


        # Fit model
        classifier, model_info.time_training = fit_model(classifier, fold.train_x, fold.train_y)

        # Make predictions
        if DO_PREDICTIONS:
            y_predict, model_info.time_prediction = predict(classifier, fold.test_x)

            # Complete the info about evaluation
            model_info.y_predict = y_predict
            model_info.y_test = fold.test_y
            model_info.accuracy = round((accuracy_score(model_info.y_test, model_info.y_predict) * 100), 2)
            model_info.cross_validation_round = cross_validation_round

            # Info about segments
            model_info.segments_y_test = labels_fold(fold.test_y)
            model_info.segments_y_predict = ensemble_predictions(y_predict)
            model_info.segments_accuracy = round(
                (accuracy_score(model_info.segments_y_test, model_info.segments_y_predict) * 100), 2)

            # Generate output files
            Output.create_confusion_matrices(model_info)
            Output.write_summary_to_csv(model_info)
            Output.write_to_json(model_info, scale)
            Output.create_graph_alg_comparison()

        # do final evaluation using this clasifier
        if DO_FINAL_EVAL:
            print("\nDoing final evaluation...")
            ev = final_eval(classifier, cross_validation_round)
            eval_predictions.append(ev)

        cross_validation_round += 1

        if CROSS_VALIDATE == False:
            print("Cross validation disabled, quitting after the first round.")
            break

    # print confusion matrices of eval results
    if CROSS_VALIDATE and DO_FINAL_EVAL:
        compare_eval(eval_predictions[0], eval_predictions[1], 1, 2)
        compare_eval(eval_predictions[0], eval_predictions[2], 1, 3)
        compare_eval(eval_predictions[0], eval_predictions[3], 1, 4)




# Old implementation, do not use.
def evaluate_model(classifier, x, y,
                   algorithm_name, algorithm_label,
                   parameter_name, parameter_value,
                   sample_size, scale):

    # Here, information about evalutation will be stored
    model_info = ModelInfo()
    model_info.name = algorithm_name
    model_info.label = algorithm_label
    model_info.sample_size = sample_size
    model_info.param_name = parameter_name
    model_info.param_value = parameter_value

    print("")
    print("==============================================================")
    print("Evaluating model:  " + algorithm_name)
    print("With parameter:    " + parameter_name + " = " + str(parameter_value))
    print("==============================================================")

    # x = np.array(x)
    # y = np.array(y)

    # fold data to prevent cutting segments
    x_folded = data_fold(x)
    y_folded = labels_fold(y)

    x_folded = np.array(x_folded)
    y_folded = np.array(y_folded)

    rng = np.random.RandomState(31337)
    cross_validation_round = 1

    kf = KFold(n_splits=4, shuffle=True, random_state=rng)
    for train_index, test_index in kf.split(x_folded):

        print("")
        print("Cross validation round " + str(cross_validation_round))
        print("========================")



        x_train_folded, x_test_folded = x_folded[train_index], x_folded[test_index]
        y_train_folded, y_test_folded = y_folded[train_index], y_folded[test_index]

        # Unfold the data back to normal state
        x_train, x_test =   data_unfold(x_train_folded),   data_unfold(x_test_folded)
        y_train, y_test = labels_unfold(y_train_folded), labels_unfold(y_test_folded)

        # Fit model
        classifier, model_info.time_training = fit_model(classifier, x_train, y_train)

        # Make predictions
        y_predict, model_info.time_prediction = predict(classifier, x_test)

        # Complete the info about evaluation
        model_info.y_predict = y_predict
        model_info.y_test = y_test
        model_info.accuracy = round((accuracy_score(model_info.y_test, model_info.y_predict) * 100), 2)
        model_info.cross_validation_round = cross_validation_round

        # Info about segments
        model_info.segments_y_test = y_test_folded
        model_info.segments_y_predict = ensemble_predictions(y_predict)
        model_info.segments_accuracy = round((accuracy_score(model_info.segments_y_test, model_info.segments_y_predict) * 100), 2)

        cross_validation_round += 1

        # Generate output files
        Output.create_confusion_matrices(model_info)
        Output.write_summary_to_csv(model_info)
        Output.write_to_json(model_info,scale)
        Output.create_graph_alg_comparison()

        if CROSS_VALIDATE == False:
            print("Cross validation disabled, quitting after the first round.")
            break






