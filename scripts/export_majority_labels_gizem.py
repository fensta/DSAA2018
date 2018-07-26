"""
Extract majority labels of Trump training set for Gizem, so that she can
investigate the effect of low agreement tweets on classifier performance by
merging the training set with the crowdsourced labels
"""
import os
import unicodecsv as csv

from prepare_datasets_for_agreement_experiment_fixed import \
    get_agreement_and_sentiment_labels_training, read_json


# Name under which worker agreement ("low" or "high") is stored
LABEL_CLA = "agreement"
SENTIMENT_LABEL = "sentiment_label"
RELEVANCE_LABEL = "relevance_label"
TEXT = "text"


def get_train_agreement(train_path):
    """
    Computes "low" or "high" agreement for each tweet in the training set.

    Parameters
    ----------
    train_path: str - path to training set in json format.

    Returns
    -------
    dict.
    {tid:
        {
            <LABEL_CLA>: agreement_label,
            <SENTIMENT_LABEL>: "..."    # pos, neg, neu or empty if irrelevant
            <RELEVANCE_LABEL>: "..."    # relevant or irrelevant
            <TEXT>: "..."               # Text
        }
    }

    """
    tweets = read_json(train_path)
    data = get_agreement_and_sentiment_labels_training(tweets)
    # Add text
    for tid in data:
        data[tid][TEXT] = tweets[tid][TEXT]
    return data


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))

    # Training set used to train the agreement classifier
    TRAIN = os.path.join(base_dir, "results", "dataset_twitter",
                         "dataset_twitter_watson_rosette_cleaned.json")
    # Path where result will be stored
    DST = os.path.join(base_dir, "results", "dataset_twitter",
                       "train_twitter.csv")

    tweets = get_train_agreement(TRAIN)

    # Store in tab-separated csv: tid, label, text
    with open(DST, "wb") as f:
        writer = csv.writer(f, dialect="excel", encoding="utf-8",
                            delimiter="\t")
        for tid in tweets:
            text = tweets[tid][TEXT]
            sent_label = tweets[tid][SENTIMENT_LABEL]
            # If it's not "irrelevant"
            if len(sent_label) > 0:
                label = tweets[tid][SENTIMENT_LABEL]
            else:
                label = tweets[tid][RELEVANCE_LABEL]
            agree_label = tweets[tid][LABEL_CLA]
            writer.writerow([tid, label, agree_label, text])
