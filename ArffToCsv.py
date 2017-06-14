
import os
import arff
import numpy as np
import pickle
import time
from sklearn.model_selection import train_test_split


# -------------
# Configuration
# -------------

RELOAD_FROM_FILES           = False     # If yes, data will be loaded from source files. If false, data will be loaded from serialized files to speed up the process.

# File paths:
PATH_FILES                  = '../data/'                    # Where the weka files are
PATH_DUMP_DATA              = '../serialized_data/data'     # Where the serialized data should be saved / loaded from
PATH_DUMP_LABELS            = '../serialized_data/labels'   # Where the serialized labels should be saved / loaded from


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


# Load or deserialize files
# -------------------------

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


data = []
count = 0
for file_name in files:
    # print progress
    count += 1
    if count % 100 == 0:
        print('%s of %s' % (count, len(files)))

    for mfcc in arff.load(open(PATH_FILES + file_name, mode="r"))["data"]:  # for every mfcc in a file
        data.append(list(map(str, mfcc)))  # data

print_time_stats("Loading data", start_time)


# Write data
# ---------
print('Writing data...')
start_time = time.time()

with open("dcase_2016_data.csv", "w") as fp:
    fp.write('\n'.join(list(map('\t'.join, data))) + '\n')

print_time_stats("Writing data", start_time)