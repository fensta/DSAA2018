"""
Analyze Weka predictions w.r.t. derived disagreement labels (<=50% votes for
majority label -> disagreement, else no disagreement) and ground truth
"""
import os
import csv


PREDICTED = "predicted"
ACTUAL = "actual"
PROBABILITY = "proba"
MAN_DIFFICULTY = "man_diff_label"
DEF_DIFFICULTY = "disagreement_diff_label"


def read_weka_predictions(src):
    """
    Extract Weka predictions from a csv file.
    File contains the following information:
    instance number, actual, predicted, error, prediction, tweet ID

    Parameters
    ----------
    src: str - path to csv file with predictions for all tweets.

    Returns
    -------
    dict.
    {
        tid:
            {
                <PREDICTED>: ...,
                <ACTUAL>: ...,
                <PROBABILITY>:... # probability for prediction
            }

    }

    """
    # IMPORTANT: in the .arff files "low" refers to low agreement, i.e. high
    # disagreement. Similarly, "high" refers to high agreement, i.e. low
    # disagreement. To make the code comply with the paper, we invert the labels
    # here to avoid mindfuckery.
    transl = {
        "low": "high",
        "high": "low"
    }
    tweets = {}
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=",", dialect="excel")
        for idx, row in enumerate(reader):
            # Skip header and avoid empty last line
            if idx > 0 and len(row) > 0:
                _, actual, predicted, _, proba, tid = \
                    row[0], row[1], row[2], row[3], float(row[4]), row[5]
                # Remove numbers in front of labels and reverse labels
                actual = transl[actual.split(":", 1)[1]]
                predicted = transl[predicted.split(":", 1)[1]]
                print "tid: {} actual: {} predicted: {}".format(tid, actual,
                                                                predicted)
                tweets[tid] = {}
                tweets[tid][PREDICTED] = predicted
                tweets[tid][ACTUAL] = actual
                tweets[tid][PROBABILITY] = proba
    return tweets


def read_excel_tweets(src):
    """
    Reads in tweets that were manually labeled w.r.t. difficulty in Excel.

    Parameters
    ----------
    src: str - path where csv file is stored.

    Returns
    -------
    dict.
    Tweets.
    {tid:
        {
            <MAN_DIFFICULTY>: manual difficulty label,  # {A, E, I, B, O}
            <DEF_DIFFICULTY>: definition difficulty label,  # {LD, HD}
        }
    }

    """
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=";", dialect="excel")
        tweets = {}
        uniq = set()
        for idx, row in enumerate(reader):
            # tid, text, majority label, definition difficulty, manual
            # difficulty, explanation
            tid, _, _, def_diff, man_diff, explain = \
                row[0], row[1], row[2], row[3], row[4], row[5]
            # print tid, def_diff, man_diff
            # Some labels have whitespaces
            man_diff = man_diff.strip()
            # Sanity check - label exists for each tweet
            assert(len(man_diff) > 0)
            assert(len(def_diff) == 2)
            # Store info for tweet
            tweets[tid] = {}
            tweets[tid][MAN_DIFFICULTY] = man_diff
            tweets[tid][DEF_DIFFICULTY] = def_diff
            uniq.add(man_diff)
    return tweets


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    FIG_DIR = os.path.join(base_dir, "results", "figures")
    PREDICT_DIR = os.path.join(base_dir, "results", "predictions")
    src = os.path.join(PREDICT_DIR, "test.csv")

    # 1. Get predictions F
    predictions = read_weka_predictions(src)

    # 2. Get disagreement labels D and ground truth G
    src = os.path.join(base_dir, "results", "ambiguous_difficult",
                       "difficulty_labels_all_votes_correct_labeled.csv")
    tweets = read_excel_tweets(src)


    # 3. Create confusion matrix as follows:
    #   * if D=high, F = high, G=high (=A, B, I, or O) -> TP
    #   * if D=high, F=low, G=high (=A,B,I, or O) -> FN
    #   * if D=low, F=high, G=low (=E) -> FP
    #   * if D=low, F=low, G=low (=E) -> TN
    #   * if D=low, F=high, G=high (=A, B, I, or O) -> TP (but we need to
    #     analyze why D=low)
    #   * if D=high, F=low, G=low (=E) -> TN (but we need to analyze why
    #     D=high, probably due to low-quality labels)
    #   * if D=low, F=low, G=high (=A,B,I, or O) -> FN (but we need to analyze
    #     why D=low)
    #   * if D=high, F=high, G=low (=E) -> FP (but we need to analyze why
    #     D=high)

