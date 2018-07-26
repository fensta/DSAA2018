"""
Creates csv file that contains all tweets, their majority label, if there's
high disagreement for them or not (<= 50% of votes for majority label)

tweet id, text, majority label, high disagreement (or not)

"""
import os
import unicodecsv as csv
from collections import Counter


# Default label assigned by annotator - it indicates no label was assigned
# for a certain hierarchy level, e.g. ["Relevant", "Factual", "?"] means
# on 1st level "Relevant" was assigned, on 2nd level "Factual", hence the
# 3rd level wasn't displayed anymore, so "?" is used
EMPTY = "?"
# Default value for annotation time
ZERO = 0

# Keys for dictionary
TEXT = "text"
LABEL = "label"
TIME = "time"
TIMES = "times"
TID = "tid"
UID = "uid"
WID = "wid"
VOTES = "votes"
ORDERED = "ordered"
LABELED = "labeled"
TYPE = "ds_type"
MAJORITY_LABEL = "maj_label"
DISAGREEMENT = "disag"
DIFFICULTY = "man_diff_label"
EXPLANATION = "explain"

# Labels to use for majority labels
# I think your suggestion for deriving sentiment is fine. That means for
# our existing labels from the previous labeling experiments in Magdeburg and
# Sabanci we only consider sentiment of relevant tweets and otherwise declare
# them irrelevant.
GROUND_TRUTH = {
    "Irrelevant": "Irrelevant",
    "Relevant": "Relevant",
    "Factual": "Neutral",
    "Non-factual": "Negative",  # Will be replaced anyway by sentiment label
    "Positive": "Positive",
    "Negative": "Negative",
    EMPTY: EMPTY    # Only added to make the code work - used for placeholders
}

# Since the AMT project was copied from another experiment, we forgot to
# change the names of the radio buttons in the new layout, so the old labels
# from the previous TREC dataset experiment
# (not relevant, relevant, highly relevant, I can't judge) were used
# Thus, we need to rename them manually.
LABEL_MAPPING = {
    "Not Relevant": "Positive",
    "Relevant": "Neutral",
    "Highly Relevant": "Negative",
    "I can't judge": "Irrelevant"
}


def store_tweets_csv(tweets, dst):
    """
    Stores the tweets in a csv file separated by commas.

    Output format of columns:
    tweet id, text, majority label, has high disagreement? (LD vs. HD),
    manual difficulty label, explanation for difficulty label)

    Parameters
    ----------
    tweets: dict - tweets.
    dst: str - path where the file will be stored.

    """
    with open(dst, "wb") as f:
        writer = csv.writer(f, dialect='excel', encoding='utf-8',
                            delimiter=",")
        for tid in tweets:
            # Tweet text
            text = tweets[tid][TEXT]
            maj = tweets[tid][MAJORITY_LABEL]
            dis = tweets[tid][DISAGREEMENT]
            diff = tweets[tid][DIFFICULTY]
            expl = tweets[tid][EXPLANATION]

            # Add dataset name to which tweet belongs
            data = [tid, text, maj, dis, diff, expl]
            writer.writerow(data)


def read_tweets_from_csv(src, is_flat):
    """
    Reads in tweets from dataset in csv format separated by commas.

    Parameters
    ----------
    src: str - path where csv file is stored.
    is_flat: bool - True if <src> uses the flat labeling scheme. Otherwise it's
    assumed that the hierarchical one is used.

    Returns
    -------
    dict.
    Tweets.
    {tid:
        {
            <VOTES>: ...., # Number of workers who labeled the tweet
            # Annotation time of i-th annotator for j-th hierarchy level
            <TIMES>: [[time1 by anno1, time2 by anno1, time3 by anno1], [..]]
            <LABEL>: [[label1 by anno1, label2 by anno1, label3 by anno1], [..]]
            <UID>: .... # ID of tweet author (to retrieve tweets from Twitter)
            <WID>: [ID of anno1 who labeled this tweet, ID of anno2...]
            <TYPE>: ...
            <TEXT>: ... # Tweet message
        }
    }

    """
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        tweets = {}
        for row in reader:
            tid, uid, ds_name, text, votes, rest = row[0], row[1], row[2], \
                                                   row[3], int(row[4]), row[5:]
            labels = []
            times = []
            if is_flat:
                wids, labelss, timess = rest[:votes], rest[votes: 2*votes], \
                                        rest[2*votes:]
                # 1 annotation time per worker
                for t in timess:
                    times.append(float(t))
                # 1 label per worker
                for l in labelss:
                    labels.append(l)
            else:
                wids, labelss, timess = rest[:votes], \
                    rest[votes: 3*votes+votes], rest[3*votes+votes:]

                # 3 labels per worker
                for i in xrange(0, len(labelss), 3):
                    one, two, three = labelss[i], labelss[i+1], labelss[i+2]
                    labels.append([one, two, three])
                # 3 annotation times per worker

                for i in xrange(0, len(timess), 3):
                    one, two, three = float(timess[i]), float(timess[i+1]), \
                                      float(timess[i+2])
                    times.append([one, two, three])
            # Store info for tweet
            tweets[tid] = {}
            tweets[tid][VOTES] = votes
            tweets[tid][LABEL] = labels
            tweets[tid][WID] = wids
            tweets[tid][UID] = uid
            tweets[tid][TIMES] = times
            tweets[tid][TYPE] = ds_name
            tweets[tid][TEXT] = text
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
            <DIFFICULTY>: difficulty label
            <EXPLANATION>: explanation of <DIFFICULTY>
        }
    }

    """
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=";", dialect="excel")
        tweets = {}
        for idx, row in enumerate(reader):
            # Skip header
            if idx > 0:
                tid, _, diff, explain = row[0], row[1], row[2], row[3]
                print tid, diff, explain

                # Store info for tweet
                tweets[tid] = {}
                tweets[tid][EXPLANATION] = explain
                tweets[tid][DIFFICULTY] = diff
    return tweets


def extract_tweets(tweets, existing_labels):
    """
    Extracts the tweets with label disagreement according to their first k votes
    that workers assigned. If there are n votes for a tweet and less than n/2+1
    votes were cast on the majority label, that tweet is considered ambiguous.

    Parameters
    ----------
    tweets: dict - dictionary of tweets from the dataset.
    existing_labels: dict - dictionary of tweets which have been already labeled
    manually w.r.t. difficulty.

    Returns
    -------
    dict.
    {tid:
         {
            <TEXT>: tweet text,
            <MAJORITY_LABEL>: majority label,
            <DISAGREEMENT>: HD or LD    # high or low disagreement
            <DIFFICULTY>: empty or assigned labels # not empty if it was
                                                   # already labeled
                                                   # somewhere else
            <EXPLANATION>: emtpy or explanation for <DIFFICULTY>
         }
    }
    """
    res = {}
    chk = 0
    for tid in tweets:
        labels = tweets[tid][LABEL]

        # Consider only first k votes
        distrib = Counter(labels)
        total_votes_per_tweet = sum(distrib.values())
        majority_thresh = 1.0 * total_votes_per_tweet / 2
        majority, count = distrib.most_common()[0]
        print "{} majority: {} count: {}".format(labels, majority, count)
        disag = "LD"    # Low disagreement
        if count <= majority_thresh:
            disag = "HD"
        res[tid] = {}
        res[tid][TEXT] = tweets[tid][TEXT]
        res[tid][MAJORITY_LABEL] = majority
        res[tid][DISAGREEMENT] = disag
        res[tid][DIFFICULTY] = ""
        res[tid][EXPLANATION] = ""
        if tid in existing_labels:
            res[tid][DIFFICULTY] = existing_labels[tid][DIFFICULTY]
            res[tid][EXPLANATION] = existing_labels[tid][EXPLANATION]
            chk += 1
        print res[tid]
    print "existing labels:", chk
    return res


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Read in manually labeled (w.r.t. difficulty) tweets that were labeled
    # using Excel
    src = os.path.join(base_dir, "results", "ambiguous_difficult",
                       "ambiguous_merged_with_easy_label.csv")
    excel_tweets = read_excel_tweets(src)
    print "annotated in excel:", len(excel_tweets)

    # Choose all tweets from TRAIN + LOW + MEDIUM + HIGH
    src = os.path.join(base_dir, "results", "export",
                       "combined_tweets.csv")
    dst_dir = os.path.join(base_dir, "results", "ambiguous_difficult")
    if not os.path.exists(src):
        raise Exception("Wrong path to dataset")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    tweets = read_tweets_from_csv(src, True)

    # Use all available votes per tweet
    dst = os.path.join(dst_dir, "difficulty_labels_all_votes.csv")
    new_tweets = extract_tweets(tweets, excel_tweets)
    print "tweets: {}".format(len(new_tweets))
    store_tweets_csv(new_tweets, dst)
