"""
Contains a collection of functions to read prediction outputs from Weka.
"""
import os
import csv


def read_weka_predictions(src):
    """
    Reads the data from a Weka prediction output file and stores the
    assigned label per instance as well as the classifier prediction
    probabilities. It's assumed that there the classification task had 3
    classes.

    Parameters
    ----------
    src: str - path to Weka prediction file.

    Returns
    -------
    dict.
    Dictionary containing for each tweet ID the most likely predicted label
    and the label distribution.
    {
        tid:
        {
            "label": "bla"
            "probas: [0.3, 0.6, 0.1]
        }
    }

    """
    preds = {}
    with open(src, "rb") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for i, row in enumerate(csv_reader):
            # Skip header and empty rows (only last row should be empty)
            if i > 0 and len(row) > 0:
                inst, actual, predicted, error, prob_label1, prob_label2, \
                    prob_label3, tid = row
                # Get the name of the predicted label and cut-off the value
                # used internally inside Weka to represent it
                predicted = predicted.split(":")[1]
                # Remove trailing '*' in probabilities and convert to float
                if "*" in prob_label1:
                    prob_label1 = float(prob_label1.split("*")[1])
                else:
                    prob_label1 = float(prob_label1)
                if "*" in prob_label2:
                    prob_label2 = float(prob_label2.split("*")[1])
                else:
                    prob_label2 = float(prob_label2)
                if "*" in prob_label3:
                    prob_label3 = float(prob_label3.split("*")[1])
                else:
                    prob_label3 = float(prob_label3)
                preds[tid] = {
                    "label": predicted,
                    # Order of labels: low, medium, high
                    "probas": [prob_label1, prob_label2, prob_label3]
                }
    return preds


def read_weka_predictions_binary(src):
    """
    Reads the data from a Weka prediction output file and stores the
    assigned label per instance as well as the classifier prediction
    probabilities. It's assumed that there the classification task had 3
    classes.
    Used for binary classification problems

    Parameters
    ----------
    src: str - path to Weka prediction file.

    Returns
    -------
    dict.
    Dictionary containing for each tweet ID the most likely predicted label
    and the label distribution.
    {
        tid:
        {
            "label": "bla"
            "probas: [0.3, 0.6, 0.1]
        }
    }

    """
    preds = {}
    with open(src, "rb") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for i, row in enumerate(csv_reader):
            # Skip header and empty rows (only last row should be empty)
            if i > 0 and len(row) > 0:
                inst, actual, predicted, error, prob_label1, prob_label2,\
                tid = row
                # Get the name of the predicted label and cut-off the value
                # used internally inside Weka to represent it
                predicted = predicted.split(":")[1]
                # Remove trailing '*' in probabilities and convert to float
                if "*" in prob_label1:
                    prob_label1 = float(prob_label1.split("*")[1])
                else:
                    prob_label1 = float(prob_label1)
                if "*" in prob_label2:
                    prob_label2 = float(prob_label2.split("*")[1])
                else:
                    prob_label2 = float(prob_label2)
                preds[tid] = {
                    "label": predicted,
                    # Order of labels: low, medium, high
                    "probas": [prob_label1, prob_label2]
                }
    return preds


def get_top_k_weka_predictions(src, k, binary=False):
    """
    Convenience function to return per instance not only its label, but also
    the classifier's k most certain class predictions for the instance.
    Also returns the margin between most and 2nd most certain class for an
    instance.

    Parameters
    ----------
    src: str - path to Weka prediction file.
    k: int - number of most certain classifier probabilities to return. Must be
    at least 2.
    binary: bool: True if it was a binary classification task. Else it's assumed
    that 3 class labels exist.

    Returns
    -------
    dict.
    Dictionary containing for each tweet ID the most likely predicted label
    and the label distribution.
    {
        tid:
        {
            "label": "bla"
            "probas": [0.3, 0.6, 0.1]   # All probabilities
            "top_k_probas": [0.6, 0.3]    # k best probabilities
            "margin": 0.3
        }
    }
    """
    if k < 2:
        raise ValueError("k must be >= 2")
    if binary:
        preds_ = read_weka_predictions_binary(src)
    else:
        preds_ = read_weka_predictions(src)
    # Keep only the k most certain class predictions per instance
    preds = {}
    for tid in preds_:
        # Get k most certain class predictions for the instance
        k_best_probas = sorted(preds_[tid]["probas"])[-k:]
        margin = k_best_probas[-1] - k_best_probas[-2]
        preds[tid] = {
            "label": preds_[tid]["label"],
            "probas": preds_[tid]["probas"],
            "top_k_probas": k_best_probas,
            "margin": margin
        }
    return preds


if __name__ == "__main__":
    WEKA_DIR = "/media/data/Workspaces/PythonWorkspace/phd/Analyze-Labeled-Dataset/www2018_results/weka_predictions/"
    fname = "ttt"
    src = os.path.join(WEKA_DIR, fname)
    print read_weka_predictions(src)
    k = 2
    print get_top_k_weka_predictions(src, k)
