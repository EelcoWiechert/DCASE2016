

from DataLoader import *
from Prediction import *
from Output import *

PATH_EVAL_DATA          = "../eval_obfuscated/"
RELOAD_EVAL_DATA        = False
PATH_DUMP_EVAL          = '../serialized_data/eval.p'
PATH_DUMP_EVAL_FILES    = '../serialized_data/eval_files.p'


def load_eval_data():
    if RELOAD_EVAL_DATA:
        # Load file names
        # ---------------
        print('Loading file names...')
        start_time = time.time()

        files = []
        for filename in os.listdir(PATH_EVAL_DATA):
            files.append(filename)



        print('Loading (and dumping) files...')
        start_time = time.time()
        (x, y) = load_contents(files, PATH_EVAL_DATA)

        pickle.dump(files,  open(PATH_DUMP_EVAL_FILES, "wb"), protocol=2)
        pickle.dump(x,      open(PATH_DUMP_EVAL, "wb"), protocol=2)

        print_time_stats("Loading (and dumping) files", start_time)
    else:
        files   = pickle.load(open(PATH_DUMP_EVAL_FILES, "rb"))
        x       = pickle.load(open(PATH_DUMP_EVAL, "rb"))

    return (files, x)

def eval_predict(clf, eval_data):
    (frames_y_pred, time_prediction) = predict(clf, eval_data)
    segments_y_pred = ensemble_predictions(frames_y_pred)
    return segments_y_pred

def output_eval(segment_y_pred, files, cv_round):
    with open(PATH_OUTPUT + "final_eval_cv" + str(cv_round) + ".txt", "w") as fp:
        for i in range(0, len(files)):
            fp.write(files[i] + "\t" + segment_y_pred[i][0] + "\n")

def final_eval(clf, cv_round):
    (files, x) = load_eval_data()
    segment_y_pred = eval_predict(clf, x)
    output_eval(segment_y_pred, files, cv_round)
    return segment_y_pred

def compare_eval(y_pred_1, y_pred_2, cv1, cv2):
    modelInfo = ModelInfo()
    modelInfo.name = str(cv1) + "vs" + str(cv2)
    create_confusion_matrix(modelInfo, y_pred_1, y_pred_2, "eval_comparison")
    return



