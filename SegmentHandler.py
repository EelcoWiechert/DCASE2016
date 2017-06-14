

# This file implements handling the full 30 s segments.

def data_fold(data):
    return [data[x:x + 1500] for x in range(0, len(data), 1500)]

def data_unfold(data):
    unfolded_data = []
    list(map(lambda segment: unfolded_data.extend(segment), data))
    return unfolded_data

def labels_fold(labels):
    return [labels[x] for x in range(0, len(labels), 1500)]

def labels_unfold(labels):
    multiplied_labels = list(map(lambda label: [label] * 1500, labels))
    unfolded_labels = []
    list(map(unfolded_labels.extend, multiplied_labels))
    return unfolded_labels


# For testing that folding and unfolding works correctly
def test_setup():
    start_time = time.time()
    x, y = load_data(standardize=False, normalize=False, sample_size=1)
    x = np.array(x)
    y = np.array(y)

    start_time = time.time()
    x_folded = data_fold(x)
    print_time_stats("Folded data", start_time)

    start_time = time.time()
    x_unfolded = data_unfold(x_folded)
    print_time_stats("Unfolded data", start_time)

    start_time = time.time()
    y_folded = labels_fold(y)
    print_time_stats("Folded labels", start_time)

    start_time = time.time()
    y_unfolded = labels_unfold(y_folded)
    print_time_stats("Unfolded labels", start_time)