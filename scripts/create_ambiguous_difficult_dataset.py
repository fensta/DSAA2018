"""
Creates csv files that contain only tweets with high disagreement when
considering the first k votes of workers. Since we only used HIGH in Q4 (as it
contains 8 votes per tweet), we vary k from 4...8 for dataset HIGH.

First,

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
    tweet id, user ID (on Twitter), dataset name, tweet text,
    #workers who labeled this
    tweet (let's call it n), n worker IDs, x labels assigned by n workers,
    x annotation times, where x could be 3n (if hierarchical scheme) or n (if
    flat scheme)

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
            text = tweets[tid]

            # Add dataset name to which tweet belongs
            data = [tid, text]
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


def extract_ambiguous_tweets(tweets, k, tids):
    """
    Extracts the tweets with label disagreement according to their first k votes
    that workers assigned. If there are n votes for a tweet and less than n/2+1
    votes were cast on the majority label, that tweet is considered ambiguous.

    Parameters
    ----------
    tweets: dict - dictionary of tweets from the dataset.
    k: int - only the first k votes of a tweet are considered for computing
    label disagreement.
    tids: dict - dictionary with tweet IDs that should be ignored.

    Returns
    -------
    dict.
    Dictionary containing all ambiguous/difficult tweets from a dataset that
    aren't included in <tids>.

    """
    # {tid: tweet message}
    res = {}
    # 8 workers labeled each tweet independently in HIGH
    total_votes_per_tweet = k
    majority_thresh = 1.0 * total_votes_per_tweet / 2
    for tid in tweets:
        labels = tweets[tid][LABEL]

        # Consider only first k votes
        distrib = Counter(labels[:k])
        majority, count = distrib.most_common()[0]
        # print "{} k: {} majority: {} count: {}".format(labels, labels[:k],
        #                                                majority, count)
        if count <= majority_thresh and tid not in tids:
            res[tid] = tweets[tid][TEXT]
    return res


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # HIGH dataset
    src = os.path.join(base_dir, "results", "export",
                       "high_disagreement_tweets.csv")
    dst_dir = os.path.join(base_dir, "results", "ambiguous_difficult")
    if not os.path.exists(src):
        raise Exception("Wrong path to HIGH dataset")
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    # Number of votes to consider for disagreement
    ks = [8, 7, 6, 5, 4]
    tweets = read_tweets_from_csv(src, True)

    # Tweets that are already contained in 1 dataset - they shouldn't be
    # included another time
    # {tid: text}
    unique_tweets = {}
    for k in ks:
        dst = os.path.join(dst_dir, "ambiguous_difficult_first{}_votes.csv"
                           .format(k))
        new_tweets = extract_ambiguous_tweets(tweets, k, unique_tweets)
        print "#ambiguous tweets in HIGH using first {} votes: {}"\
            .format(k, len(new_tweets))
        unique_tweets.update(new_tweets)
        store_tweets_csv(new_tweets, dst)

    dst = os.path.join(dst_dir, "ambiguous_difficult_combined.csv")
    store_tweets_csv(unique_tweets, dst)

    print "#ambiguous tweets", len(unique_tweets)
    for tid in unique_tweets:
        print tid

