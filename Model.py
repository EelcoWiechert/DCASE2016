
# Class to hold information about the algorithm performace
class ModelInfo:
    name = ""
    label = ""
    param_name = ""
    param_value = ""
    time_training = -1
    time_prediction = -1
    y_test = []
    y_predict = []
    segments_y_test = []
    segments_y_predict = []
    sample_size = -1
    cross_validation_round = -1
    accuracy = -1
    segments_accuracy = -1
    recall = -1


class Fold:
    train_x = []
    test_x = []
    eval_x = []
    train_y = []
    test_y = []
    eval_y = []