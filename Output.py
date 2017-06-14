
## This file

import os.path
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import json
import numpy as np
from Model import ModelInfo
from sklearn.metrics import confusion_matrix
from confusion_matrix import plot_confusion_matrix

# File paths:
PATH_OUTPUT                 = '../output/'                  # Where you want to store the images
PATH_SUMMARY_FILE           = PATH_OUTPUT + "performance_summary.csv"


def write_summary_to_csv(model_info: ModelInfo):
    print("Writing to CSV summary file...")

    # if file does not exist, create it and write the header there
    if os.path.isfile(PATH_SUMMARY_FILE) == False:
        with open(PATH_SUMMARY_FILE, "w") as fp:
            header = [
                "algorithm_name",
                "label",
                "param_name",
                "param_value",
                "CV_round",
                "sample_size",
                "training_time",
                "prediction_time",
                "frame_accuracy",
                "segment_accuracy"
            ]
            fp.write('\t'.join(header) + "\n")

    with open(PATH_SUMMARY_FILE, "a") as fp:
        column_values = [
            model_info.name,
            model_info.label,
            model_info.param_name,
            str(model_info.param_value),
            str(model_info.cross_validation_round),
            str(model_info.sample_size),
            str(model_info.time_training),
            str(model_info.time_prediction),
            str(model_info.accuracy),
            str(model_info.segments_accuracy)
        ]
        fp.write('\t'.join(column_values) + "\n")


# Create one CM for frame prediction, one for ensembled 30 s segment prediction
def create_confusion_matrices(model_info: ModelInfo):
    create_confusion_matrix(model_info, model_info.y_test, model_info.y_predict, "frame")
    create_confusion_matrix(model_info, model_info.segments_y_test, model_info.segments_y_predict, "segment")


def create_confusion_matrix(model_info, y_test, y_predict, name):
    # Creating confusion matrix
    # -------------------------
    print('Creating confusion matrix...')

    # Get a list of valid labels
    labels = np.unique(y_test)

    cnf_matrix = confusion_matrix(y_test, y_predict)
    np.set_printoptions(precision=2)

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=labels, title='Confusion matrix, without normalization')

    filename = PATH_OUTPUT + name + "_" + model_info.name + "-" + model_info.label + "-" + \
               model_info.param_name + "-" + str(model_info.param_value) + "_" + \
               "cv" + str(model_info.cross_validation_round )+ '-confusion_matrix.pdf'
    plt.savefig(filename, format ='pdf', dpi=300)
    plt.close()

# Create a graph comparing all the algorithms
def create_graph_alg_comparison():
    # Create accuracy graphs
    # ----------------------

    with open(PATH_OUTPUT + 'performance_metrics') as json_file:
        performance_dic = json.load(json_file)

    # Create accuracy plot
    x_acc = []
    y_acc_normal = []
    y_acc_normalized = []
    y_acc_standardized = []

    for classifier, performances in performance_dic.items():
        if classifier not in x_acc:
            x_acc.append(classifier)
            try:
                y_acc_normal.append(performances['normal']['accuracy'])
            except:
                y_acc_normal.append(0)
            try:
                y_acc_normalized.append(performances['normalized']['accuracy'])
            except:
                y_acc_normalized.append(0)
            try:
                y_acc_standardized.append(performances['standardized']['accuracy'])
            except:
                y_acc_standardized.append(0)

    y_pos = np.arange(len(x_acc))
    plt.xticks((y_pos+1)/2, x_acc)
    if len(y_acc_normal)>0:
        plt.bar(y_pos, y_acc_normal, width=0.5, color="blue",edgecolor="black", label='normal')
    if len(y_acc_normal)>0:
        plt.bar(y_pos + 0.5, y_acc_normalized, width=0.5, color="red", edgecolor="black", label='normalized')
    if len(y_acc_normal)>0:
        plt.bar(y_pos + 1, y_acc_standardized, width=0.5, color="green", edgecolor="black", label='standardized')
    plt.xlabel('Classifiers')
    plt.ylabel('Accuracy in percentage')
    plt.legend()
    plt.title('Accuracy of different classifiers')
    plt.savefig(PATH_OUTPUT + 'accuracy_graph', dpi=1000)


def write_to_json(model_info, scale):
    # Writing to JSON
    # ----------------
    print('Write parameters to JSON file...')

    accuracy = round((accuracy_score(model_info.y_test, model_info.y_predict) * 100), 2)

    try:
        with open(PATH_OUTPUT + 'performance_metrics') as json_file:
            performance_dic = json.load(json_file)
        if model_info.name not in performance_dic:
            performance_dic[model_info.name] = {}
        if scale not in performance_dic[model_info.name]:
            performance_dic[model_info.name][scale] = {}
        performance_dic[model_info.name][scale]['accuracy'] = accuracy
        performance_dic[model_info.name][scale]['sample_size'] = model_info.sample_size
        performance_dic[model_info.name][scale]['prediction_time'] = model_info.time_prediction
        performance_dic[model_info.name][scale]['training_time'] = model_info.time_training

        with open(PATH_OUTPUT + 'performance_metrics', 'w') as outfile:
            json.dump(performance_dic, outfile)

    except:
        performance_dic = {}
        performance_dic[model_info.name] = {}
        performance_dic[model_info.name][scale] = {}
        performance_dic[model_info.name][scale]['accuracy'] = accuracy
        performance_dic[model_info.name][scale]['sample_size'] = model_info.sample_size
        performance_dic[model_info.name][scale]['prediction_time'] = model_info.time_prediction
        performance_dic[model_info.name][scale]['training_time'] = model_info.time_training

        with open(PATH_OUTPUT + 'performance_metrics', 'w') as outfile:
            json.dump(performance_dic, outfile)