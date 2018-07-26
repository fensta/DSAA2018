"""
Stores all the tweets with their meta data stored in a MongoDB instance in a
json file for convenient access.
"""
import os
import random
import unicodecsv as csv
from dateutil import parser
import operator

from anno import Annotator
import utility


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
GROUND_TRUTH_DIFF = "man_diff_label"
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

SEED = 13
random.seed(SEED)


def anonymize_train_hierarchical(ds_name, anno_coll_name="user",
                                 tweet_coll_name="tweets", cleaned=False):
    """
    Anonymizes the training dataset by creating two datastructures, one storing
    tweet information, the other one information about the crowd workers.
    It stores the original hierarchical labeling scheme. Uses as labels
    "Irrelevant", "Relevant", "Neutral", "Positive", "Negative", "Non-Factual".
    For "Irrelevant" tweets no other labels will be stored, instead "?" is
    stored as a placeholder. Each worker assigns 3 labels to each labeled tweet.
    "?" is used as a placeholder if certain labels weren't chosen.

    Parameters
    ----------
    ds_name: str - name of the dataset.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.

    Returns
    -------
    dict, dict.
    Tweets, workers.
    Tweets:
    {tid:
        {
            <TEXT>: ...,
            <VOTES>: ...., # Number of workers who labeled the tweet
            # Annotation time of i-th annotator for j-th hierarchy level
            <TIMES>: [[time1 by anno1, time2 by anno1, time3 by anno1], [..]]
            <LABEL>: [[label1 by anno1, label2 by anno1, label3 by anno1], [..]]
            <UID>: .... # ID of tweet author (to retrieve tweets from Twitter)
            <WID>: [ID of anno1 who labeled this tweet, ID of anno2...]
        }
    }
    workers:
    {
        wid:
            {
            <LABELED>: ...# Number of labeled tweets by this annotator
            <ORDERED>: [tid1, tid2,...] # Order in which tweets were labeled
            }
    }


    """
    # Names of all the MongoDBs which contain data from experiments
    DB_NAMES = [
        "lturannotationtool",
        "mdlannotationtool",
        "mdsmannotationtool",
        "turannotationtool",
        "harishannotationtool",
        "kantharajuannotationtool",
        "mdslaterannotationtool",
    ]
    # Indices of DBs to consider for analyses - all DBs
    dbs = [0, 1, 2, 3, 4, 5, 6]
    annos, counts = read_dataset_hierarchical(
        DB_NAMES, dbs, ds_name, anno_coll_name=anno_coll_name,
        tweet_coll_name=tweet_coll_name, cleaned=cleaned, min_annos=3)
    tweets = aggregate_data_per_tweet(annos, counts)
    workers = {}
    for anno in annos:
        labeled = anno.tweets.ids
        workers[anno.wid] = {}
        workers[anno.wid][ORDERED] = labeled
        workers[anno.wid][TYPE] = anno.ds_name
        workers[anno.wid][LABELED] = len(labeled)
        print anno.name, len(labeled), anno.wid
    return tweets, workers


def anonymize_train_flat(ds_name, anno_coll_name="user",
                         tweet_coll_name="tweets", cleaned=False):
    """
    Anonymizes the training dataset by creating two datastructures, one storing
    tweet information, the other one information about the crowd workers.
    It stores a flat labeling scheme that is compatible with the crowdsourced
    datasets. Each annotator assigns only one label per tweet. This function
    converts the hierarchical scheme into a flat one that was used in
    crowdsourcing. The conversion is as follows:

    Relevant + Positive -> Positive
    Relevant + Negative -> Negative
    Relevant + Factual -> Neutral
    Irrelevant + ... -> Irrelevant

    Parameters
    ----------
    ds_name: str - name of the dataset.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.

    Returns
    -------
    dict, dict.
    Tweets, workers.
    Tweets:
    {tid:
        {
            <TEXT>: ...,
            <VOTES>: ...., # Number of workers who labeled the tweet
            # Annotation time of i-th annotator
            <TIMES>: [[time by anno1], [time by anno2]]
            <LABEL>: [[label by anno1], [label2 by anno2],[..]]
            <UID>: .... # ID of tweet author (to retrieve tweets from Twitter)
            <WID>: [ID of anno1 who labeled this tweet, ID of anno2...]
            <TYPE>: ...
        }
    }
    workers:
    {
        wid:
            {
            <LABELED>: ...# Number of labeled tweets by this annotator
            <TYPE> ...
            <ORDERED>: [tid1, tid2,...] # Order in which tweets were labeled
            }
    }

    """
    # Names of all the MongoDBs which contain data from experiments
    DB_NAMES = [
        "lturannotationtool",
        "mdlannotationtool",
        "mdsmannotationtool",
        "turannotationtool",
        "harishannotationtool",
        "kantharajuannotationtool",
        "mdslaterannotationtool",
    ]
    # Indices of DBs to consider for analyses - all DBs
    dbs = [0, 1, 2, 3, 4, 5, 6]
    annos, counts = read_dataset_flat(
        DB_NAMES, dbs, ds_name, anno_coll_name=anno_coll_name,
        tweet_coll_name=tweet_coll_name, cleaned=cleaned, min_annos=3)
    tweets = aggregate_data_per_tweet(annos, counts)
    workers = {}
    for anno in annos:
        labeled = anno.tweets.ids
        workers[anno.wid] = {}
        workers[anno.wid][ORDERED] = labeled
        workers[anno.wid][TYPE] = anno.ds_name
        workers[anno.wid][LABELED] = len(labeled)
        print anno.name, len(labeled), anno.wid
    return tweets, workers


def aggregate_data_per_tweet(annos, counts):
    """
    Aggregates the data from annotators to tweets.

    Parameters
    ----------
    annos: list of anno.Annotator - annotators with extracted tweets.
    counts: dict - raw counts how often each tweet was labeled. Contains only
    tweet IDs that were sufficiently often labeled, namely <min_annos> times.

    Returns
    -------
    dict.
    For each tweet ID the data to be written to file:
    {tid:
        {
            <TEXT>: ...,
            <VOTES>: ..., # Number of workers who labeled the tweet
            # Annotation time of i-th annotator for j-th hierarchy level
            <TIMES>: [[time1 by anno1, time2 by anno1, time3 by anno1], [..]]
            <LABEL>: [[label1 by anno1, label2 by anno1], label3 by anno1],[..]]
            <UID>: ... # ID of tweet author (to retrieve tweets from Twitter)
            <WID>: [ID of anno1 who labeled this tweet, ID of anno2...]
            <TYPE>: ...
            <TEXT>: ... # tweet message
        }
    }

    """
    tweets = {}
    for anno in annos:
        for idx, tid in enumerate(anno.all_tweets()):
            # New tweet that was sufficiently often labeled
            if tid in counts and tid not in tweets:
                tweets[tid] = {
                    # ID of tweet author
                    UID: anno.tweets.user_ids[idx],
                    # List of lists: inner list represents labels assigned to
                    # i-th hierarchy level by the annotator
                    LABEL: [],
                    # List of lists: inner list represents times needed
                    # for assigning i-th hierarchy level by the annotator
                    TIMES: [],
                    VOTES: counts[tid],
                    # i-th worker labeled a tweet
                    WID: [],
                    # Dataset to which tweet belongs
                    TYPE: anno.ds_name,
                    # Tweet text
                    TEXT: anno.tweets.texts[idx]
                }
            tweets[tid][LABEL].append(anno.tweets.labels[idx])
            tweets[tid][TIMES].append(anno.tweets.anno_times_list[idx])
            tweets[tid][WID].append(anno.wid)

    return tweets


def read_dataset_hierarchical(dbs, db_idxs, ds_name, anno_coll_name="user",
                              tweet_coll_name="tweets", cleaned=False,
                              min_annos=3):
    """
    Read dataset from MongoDB.

    Parameters
    ----------
    dbs: list of strings - names of the existing DBs.
    db_idxs: list of ints - name of the MongoDB from where data should be read.
    ds_name: str - name of the dataset.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.

    Returns
    --------
    list, dict.
    List of Annotator objects in institution with their tweets.
    Dictionary storing for each tweet how many annotators labeled it.

    """
    # Store a list of annotators per group/institution
    inst_annos = []
    # For anonymization assign each worker an ID
    worker_id = 0
    for db_idx in db_idxs:
        # Get DB name
        db = dbs[db_idx]
        tweet_coll, anno_coll = utility.load_tweets_annotators_from_db(
            db, tweet_coll_name, anno_coll_name)
        # For each anno
        for anno in anno_coll.find():
            username = anno["username"]
            group = anno["group"]
            # Use username + "_" + group because 2 annotators of MD
            # labeled for S and M (otherwise their entries are overridden)
            dict_key = username + "_" + group
            inst_anno = Annotator(dict_key, group, worker_id, ds_name)
            # Tweet IDs labeled by this annotator
            labeled = anno["annotated_tweets"]
            for tid in labeled:
                second_label = EMPTY
                third_label = EMPTY
                fac_time = ZERO
                opi_time = ZERO
                tweet = utility.get_tweet(tweet_coll, tid)
                # Use Twitter ID because _id differs for the same
                # tweet as it was created in multiple DBs.
                tweet_id = tweet["id_str"]
                user_id = tweet["user"]["id_str"]
                text = tweet["text"]
                first_label = tweet["relevance_label"][username]
                rel_time = tweet["relevance_time"][username]
                # Annotator labeled the 3rd set of labels as well
                # Discard remaining labels if annotator chose
                # "Irrelevant"
                # Consider other sets of labels iff either the cleaned
                # dataset should be created and the label is "relevant"
                # OR the raw dataset should be used.
                if (cleaned and first_label != "Irrelevant") or not \
                        cleaned:
                    second_label = tweet["fact_label"][username]
                    fac_time = tweet["fact_time"][username]
                    # Annotator labeled the 3rd set of labels as well
                    if username in tweet["opinion_label"]:
                        third_label = tweet["opinion_label"][username]
                        opi_time = tweet["opinion_time"][username]
                # Add annotation times and labels to annotator
                anno_times = [rel_time, fac_time, opi_time]
                anno_time = sum(anno_times)
                # Rename the labels according to what we want
                first_label = GROUND_TRUTH[first_label]
                # Rename to "Neutral"
                if second_label == "Factual":
                    second_label = GROUND_TRUTH[second_label]
                third_label = GROUND_TRUTH[third_label]
                labels = [first_label, second_label, third_label]
                inst_anno.add_tweet(tweet_id, anno_time, labels,
                                    text, user_id, anno_times)
            # Store annotator
            inst_annos.append(inst_anno)
            worker_id += 1
    # Count for each tweet how often it was labeled.  The reason for
    # NOT counting in the previous loop is that 3 annotators of MD (M) - see
    # anno.py for a detailed explanation at the top - labeled the same tweet
    # twice, so counting would be off by 1 for 3 annotators. Therefore,
    # anno.Annotator handles these exceptions and ignores the tweets that were
    # labeled a second time.
    inst_counts = count_annotators_per_tweet(inst_annos)

    # Now only keep tweets that were labeled sufficiently often by annotators
    # Create a list of tweet IDs that must be removed since they weren't labeled
    # by enough annotators
    removed_inst_tweets = [tid for tid in inst_counts if
                           inst_counts[tid] < min_annos]
    print "remove from INSTITUTION all:", len(removed_inst_tweets)

    # Delete tweets that weren't labeled by enough annotators
    for anno in inst_annos:
        anno.delete_tweets(removed_inst_tweets)

    # Test that all tweets were removed
    for anno in inst_annos:
        for tid in anno.all_tweets():
            if tid in removed_inst_tweets:
                raise Exception("can't happen")

    # Make sure that we only count tweets that were sufficiently often
    # labeled in the institution
    for tid in removed_inst_tweets:
        del inst_counts[tid]
    print "#tweets in dataset", len(inst_counts)
    print "#annos in dataset", len(inst_annos)
    return inst_annos, inst_counts


def read_dataset_flat(dbs, db_idxs, ds_name, anno_coll_name="user",
                              tweet_coll_name="tweets", cleaned=False,
                              min_annos=3):
    """
    Read dataset from MongoDB.

    Parameters
    ----------
    dbs: list of strings - names of the existing DBs.
    db_idxs: list of ints - name of the MongoDB from where data should be read.
    ds_name: str - name of the dataset.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.

    The conversion from hierarchical to flat labeling scheme works as follows:

    Relevant + Positive -> Positive
    Relevant + Negative -> Negative
    Relevant + Factual -> Neutral
    Irrelevant + ... -> Irrelevant

    Returns
    --------
    list, dict.
    List of Annotator objects in institution with their tweets.
    Dictionary storing for each tweet how many annotators labeled it.

    """
    # Store a list of annotators per group/institution
    inst_annos = []
    # For anonymization assign each worker an ID
    worker_id = 0
    for db_idx in db_idxs:
        # Get DB name
        db = dbs[db_idx]
        tweet_coll, anno_coll = utility.load_tweets_annotators_from_db(
            db, tweet_coll_name, anno_coll_name)
        # For each anno
        for anno in anno_coll.find():
            username = anno["username"]
            group = anno["group"]
            # Use username + "_" + group because 2 annotators of MD
            # labeled for S and M (otherwise their entries are overridden)
            dict_key = username + "_" + group
            inst_anno = Annotator(dict_key, group, worker_id, ds_name)
            # Tweet IDs labeled by this annotator
            labeled = anno["annotated_tweets"]
            for tid in labeled:
                second_label = EMPTY
                third_label = EMPTY
                fac_time = ZERO
                opi_time = ZERO
                tweet = utility.get_tweet(tweet_coll, tid)
                # Use Twitter ID because _id differs for the same
                # tweet as it was created in multiple DBs.
                tweet_id = tweet["id_str"]
                user_id = tweet["user"]["id_str"]
                text = tweet["text"]
                first_label = tweet["relevance_label"][username]
                rel_time = tweet["relevance_time"][username]
                # Annotator labeled the 3rd set of labels as well
                # Discard remaining labels if annotator chose
                # "Irrelevant"
                # Consider other sets of labels iff either the cleaned
                # dataset should be created and the label is "relevant"
                # OR the raw dataset should be used.
                if (cleaned and first_label != "Irrelevant") or not \
                        cleaned:
                    second_label = tweet["fact_label"][username]
                    fac_time = tweet["fact_time"][username]
                    # Annotator labeled the 3rd set of labels as well
                    if username in tweet["opinion_label"]:
                        third_label = tweet["opinion_label"][username]
                        opi_time = tweet["opinion_time"][username]
                # Compute annotation time and flat label
                # Rename flat labels according to what we want
                anno_time = sum([rel_time, fac_time, opi_time])
                label = GROUND_TRUTH["Irrelevant"]
                if first_label == "Relevant" and second_label == "Factual":
                    label = GROUND_TRUTH[second_label]
                if first_label == "Relevant" and third_label != EMPTY:
                    label = GROUND_TRUTH[third_label]
                inst_anno.add_tweet(tweet_id, anno_time, [label], text, user_id,
                                    [anno_time])
            # Store annotator
            inst_annos.append(inst_anno)
            worker_id += 1
    # Count for each tweet how often it was labeled.  The reason for
    # NOT counting in the previous loop is that 3 annotators of MD (M) - see
    # anno.py for a detailed explanation at the top - labeled the same tweet
    # twice, so counting would be off by 1 for 3 annotators. Therefore,
    # anno.Annotator handles these exceptions and ignores the tweets that were
    # labeled a second time.
    inst_counts = count_annotators_per_tweet(inst_annos)

    # Now only keep tweets that were labeled sufficiently often by annotators
    # Create a list of tweet IDs that must be removed since they weren't labeled
    # by enough annotators
    removed_inst_tweets = [tid for tid in inst_counts if
                           inst_counts[tid] < min_annos]
    print "remove from INSTITUTION all:", len(removed_inst_tweets)

    # Delete tweets that weren't labeled by enough annotators
    for anno in inst_annos:
        anno.delete_tweets(removed_inst_tweets)

    # Test that all tweets were removed
    for anno in inst_annos:
        for tid in anno.all_tweets():
            if tid in removed_inst_tweets:
                raise Exception("can't happen")

    # Make sure that we only count tweets that were sufficiently often
    # labeled in the institution
    for tid in removed_inst_tweets:
        del inst_counts[tid]
    print "#tweets in dataset", len(inst_counts)
    print "#annos in dataset", len(inst_annos)
    return inst_annos, inst_counts


def count_annotators_per_tweet(annos):
    """
    Counts how many annotators labeled each tweet.

    Parameters
    ----------
    annos: list of anno.Annotator - each annotator holds the tweets she labeled.

    Returns
    -------
    dict.
    Dictionary holding for each tweet ID  with number of annotators who
    labeled it.

    """
    # {tweet_id: count}
    counts = {}
    for anno in annos:
        for tid in anno.all_tweets():
            if tid not in counts:
                counts[tid] = 0
            counts[tid] += 1
    return counts


def store_tweets_csv(tweets, dst, is_anonymous):
    """
    Stores the anonymized tweets in a csv file separated by commas.

    Output format of columns:
    tweet id, user ID (on Twitter), dataset name, ground truth difficulty label,
    tweet text,
    #workers who labeled this
    tweet (let's call it n), n worker IDs, x labels assigned by n workers,
    x annotation times, where x could be 3n (if hierarchical scheme) or n (if
    flat scheme)

    Parameters
    ----------
    tweets: dict - tweets.
    dst: str - path where the file will be stored.
    is_anonymous: bool - True if original tweet text should be stored. Otherwise
    a placeholder text is stored.

    """
    with open(dst, "wb") as f:
        writer = csv.writer(f, dialect='excel', encoding='utf-8',
                            delimiter=",")
        for tid in tweets:
            # How many workers labeled the tweet
            votes = tweets[tid][VOTES]
            # Annotation times of all workers
            times = tweets[tid][TIMES]
            uid = tweets[tid][UID]
            # Worker IDs
            wids = tweets[tid][WID]
            labels = tweets[tid][LABEL]
            ds_name = tweets[tid][TYPE]
            # Tweet text
            text = tweets[tid][TEXT]
            # Ground truth difficulty label
            diff_label = tweets[tid][GROUND_TRUTH_DIFF]
            if not is_anonymous:
                # Include original tweet and remove newlines
                text = text.replace("\r", "").replace("\n", "")
            else:
                # Use placeholder
                text = "Download via Twitter API"

            # Add dataset name to which tweet belongs
            data = [tid, uid, ds_name, diff_label, text, votes]
            # Add worker IDs
            for w in wids:
                data.append(w)
            # Add labels
            for lvl in labels:
                for l in lvl:
                    data.append(l)
            # Add annotation times
            print "tid, ds", tid, ds_name, times
            for lvl in times:
                for t in lvl:
                    data.append(t)
            writer.writerow(data)


def store_workers_csv(workers, dst):
    """
    Stores the anonymized workers in a csv file separated by commas.

    Output format of columns:
    worker ID, dataset name, #labeled tweets (let's call it n), n columns with
    tweet IDs in the order that worker labeled them

    Parameters
    ----------
    workers: dict - workers.
    dst: str - path where the file will be stored.

    """
    with open(dst, "wb") as f:
        writer = csv.writer(f, dialect='excel', encoding='utf-8',
                            delimiter=",")
        for wid in workers:
            # How many tweets did this worker label?
            labeled = workers[wid][LABELED]
            # IDs of labeled tweets in order that worker labeled them
            ordered = workers[wid][ORDERED]
            # Dataset to which worker belongs
            ds_name = workers[wid][TYPE]
            data = [wid, ds_name, labeled]
            # Add tweet IDs
            for o in ordered:
                data.append(o)
            writer.writerow(data)


def anonymize_crowdsourced_dataset(src, tweet_dir, votes, ds_name,
                                   worker_offset=0):
    """
    Anonymizes a crowdsourced dataset stored in csv format separated by commas
    by creating two datastructures, one storing
    tweet information, the other one information about the crowd workers.
    Each annotator assigns only one label per tweet, either Positive, Negative,
    Irrelevant, or Neutral.

    Parameters
    ----------
    src: str - path to the input file.
    tweet_dir: str - directory in which the tweets are stored (to obtain
    some metadata).
    votes: int - number of votes per tweet, i.e. #workers who labeled each
    tweet.
    ds_name: str - name of the dataset.
    worker_offset: int - lowest ID to be used for workers - useful if other
    datasets will be merged to avoid having the same IDs for different workers.

    Returns
    -------
    dict, dict.
    Tweets, workers.
    Tweets:
    {tid:
        {
            <TEXT>: ...,
            <VOTES>: ...., # Number of workers who labeled the tweet
            # Annotation time of i-th annotator
            <TIMES>: [[time by anno1], [time by anno2]]
            <LABEL>: [[label by anno1], [label by anno2],[..]]
            <UID>: .... # ID of tweet author (to retrieve tweets from Twitter)
            <WID>: [ID of anno1 who labeled this tweet, ID of anno2...]
            <TYPE>: ...
        }
    }
    workers:
    {
        wid:
            {
            <LABELED>: ...# Number of labeled tweets by this annotator
            <TYPE>: ...
            <ORDERED>: [tid1, tid2,...] # Order in which tweets were labeled
            }
    }

    """
    # Column indices in AMT csv files
    wid_idx = 15
    label_idx = 30
    tid_idx = 27
    time_idx = 23
    time_start_idx = 17
    text_idx = 29

    tweets = {}
    workers = {}
    worker_id = worker_offset
    # {wid: computed_worker_id}
    wid_mapping = {}
    # {wid:
    #   {
    #       [(tid1, start_time), (tid2, start_time)]
    #   }
    # }
    unordered = {}
    with open(src, "rb") as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            # Skip header
            if idx > 0:
                tid = row[tid_idx]
                label = row[label_idx]
                label = LABEL_MAPPING[label]
                wid = row[wid_idx]
                # Find ID of a tweet's author by looking at each file
                uid = ""
                for fname in os.listdir(tweet_dir):
                    utid = fname.split("_")[1].split(".")[0]
                    if tid == utid:
                        uid = fname.split("_")[0]
                # Add a new worker
                if wid not in wid_mapping:
                    wid_mapping[wid] = worker_id
                    workers[worker_id] = {}
                    unordered[worker_id] = []
                    worker_id += 1

                # Add a new tweet
                if tid not in tweets:
                    tweets[tid] = {}
                    tweets[tid][TEXT] = row[text_idx]
                    # Store ID of tweet author
                    tweets[tid][UID] = uid
                    # Store how many workers labeled it
                    tweets[tid][VOTES] = votes
                    tweets[tid][TIMES] = []
                    tweets[tid][WID] = []
                    tweets[tid][LABEL] = []
                    tweets[tid][TYPE] = ds_name
                # Store label assigned by worker to current tweet
                tweets[tid][LABEL].append([label])
                start = parser.parse(row[time_start_idx])
                anno_time = int(row[time_idx])
                # Store annotation time of worker for current tweet
                tweets[tid][TIMES].append([anno_time])
                # Store ID of worker who assigned the label to the current tweet
                tweets[tid][WID].append(wid_mapping[wid])
                # Store which tweet was labeled starting at what time
                unordered[wid_mapping[wid]].append((tid, start))

        # Figure out in which order the tweets were labeled
        # Sort ascendingly
        for wid in unordered:
            ordered = []
            # Sort according to when worker started labeling a tweet
            res = sorted(unordered[wid], key=operator.itemgetter(1))

            print "sorted"
            print res
            # Sort other data accordingly
            for tid, start_time in res:
                ordered.append(tid)
            # Store order in which worker labeled the tweets
            workers[wid][ORDERED] = ordered
            # Store to which dataset a worker belongs
            workers[wid][TYPE] = ds_name
            # Count how many tweets each worker labeled
            workers[wid][LABELED] = len(ordered)

        print "#workers", len(workers)

        for tid in tweets:
            print tid, tweets[tid][VOTES], len(tweets[tid][TIMES]), \
                len(tweets[tid][LABEL]), tweets[tid]

            # Sanity check
            for tid in tweets:
                assert(tweets[tid][VOTES] == len(tweets[tid][TIMES]) ==
                       len(tweets[tid][LABEL]))
    return tweets, workers


def create_anonymized_combined_dataset(s0_tw, s0_wo, lo_tw, lo_wo, me_tw,
                                       me_wo, hi_tw, hi_wo, dst_tw, dst_wo):
    """
    Creates the anonymized datasets in csv format with commas separating the
    columns - one for workers, one for tweets.

    Output format for tweets:
    -------------------------
    tweet id, user ID (on Twitter), dataset name, ground truth difficulty label,
    text (placeholder, #workers who labeled this tweet (let's call
    it n), n worker IDs, n labels assigned by n workers, n annotation times.
    The order of worker IDs, labels and annotation times is the same, i.e.
    to figure out for a certain tweet which label the i-th worker assigned and
    how long the labeling took (in seconds), you can access the i-th label
    and the i-th annotation time.

    Output format for workers:
    --------------------------
    Worker ID, #tweets that worker labeled (let's call it n), n tweet IDs
    indicating in which order that worker labeled the tweets.

    Parameters
    ----------
    s0_tw: str - path where the anonymized tweets of S_0 (train) are stored in
    a csv file.
    s0_wo: str - path where the anonymized workers of S_0 (train) are stored in
    a csv file.
    lo_tw: str - path where the anonymized tweets of HIGH are stored in a csv
    file.
    lo_wo: str - path where the anonymized workers of HIGH are stored in
    a csv file.
    me_tw: str - path where the anonymized tweets of MEDIUM are stored in a csv
    file.
    me_wo: str - path where the anonymized workers of MEDIUM are stored in
    a csv file.
    hi_tw: str - path where the anonymized tweets of LOW are stored in a csv
    file.
    hi_wo: str - path where the anonymized workers of LOW are stored in
    a csv file.
    dst_tw: str - path where resulting csv for tweets should be stored.
    dst_wo: str - path where resulting csv for workers should be stored.

    """
    is_flat = True
    # Load datasets
    tweets_s0 = read_tweets_from_csv(s0_tw, is_flat)
    workers_s0 = read_workers_from_csv(s0_wo)
    tweets_lo = read_tweets_from_csv(lo_tw, is_flat)
    workers_lo = read_workers_from_csv(lo_wo)
    tweets_me = read_tweets_from_csv(me_tw, is_flat)
    workers_me = read_workers_from_csv(me_wo)
    tweets_hi = read_tweets_from_csv(hi_tw, is_flat)
    workers_hi = read_workers_from_csv(hi_wo)
    # Merge tweets
    tweets_s0.update(tweets_lo)
    tweets_s0.update(tweets_me)
    tweets_s0.update(tweets_hi)
    # Merge workers
    workers_s0.update(workers_lo)
    workers_s0.update(workers_me)
    workers_s0.update(workers_hi)

    print "#tweets", len(tweets_s0)
    print "#workers", len(workers_s0)
    #store_tweets_csv(tweets_s0, dst_tw)
    # store_tweets_csv(tweets_s0, dst_tw) can't be used because
    # read_tweets_from_csv() reads labels, annotation times not as a list
    # which was used as a hack to reuse store_tweets_csv() even for flat scheme
    with open(dst_tw, "wb") as f:
        writer = csv.writer(f, dialect='excel', encoding='utf-8',
                            delimiter=",")
        for tid in tweets_s0:
            # How many workers labeled the tweet
            votes = tweets_s0[tid][VOTES]
            # Annotation times of all workers
            times = tweets_s0[tid][TIMES]
            text = tweets_s0[tid][TEXT]
            uid = tweets_s0[tid][UID]
            # Worker IDs
            wids = tweets_s0[tid][WID]
            labels = tweets_s0[tid][LABEL]
            ds_name = tweets_s0[tid][TYPE]
            diff_label = tweets_s0[tid][GROUND_TRUTH_DIFF]

            # Add dataset name to which tweet belongs
            data = [tid, uid, ds_name, diff_label, text, votes]
            # Add worker IDs
            for w in wids:
                data.append(w)
            # Add labels
            for l in labels:
                data.append(l)
            # Add annotation times
            print "tid, ds", tid, ds_name, times
            for t in times:
                data.append(t)
            writer.writerow(data)

    store_workers_csv(workers_s0, dst_wo)


#######################################################
# Functions to read the extracted datasets into dicts #
#######################################################
def read_workers_from_csv(src):
    """
    Reads in workers from training set in csv format separated by commas.

    Parameters
    ----------
    src: str - path where csv file is stored.

    Returns
    -------
    dict.
    Workers.
    {
        wid:
            {
            <LABELED>: ...# Number of labeled tweets by this annotator
            <TYPE>: ... # Dataset name
            <ORDERED>: [tid1, tid2,...] # Order in which tweets were labeled
            }
    }

    """
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        workers = {}
        for row in reader:
            wid, ds_name, labeled, ordered = row[0], row[1], row[2], row[3:]
            workers[wid] = {}
            workers[wid][TYPE] = ds_name
            workers[wid][LABELED] = labeled
            workers[wid][ORDERED] = ordered
    return workers


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
            <GROUND_TRUTH_DIFF>: ... # Ground truth difficulty label, either
            (A)mbiguity, Lack of (B)ackground, (I)rrelevance, (O)ther, (E)asy
            (=(S)implicity in the paper)
        }
    }

    """
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter=",")
        tweets = {}
        for row in reader:
            tid, uid, ds_name, diff_label, text, votes, rest = \
                row[0], row[1], row[2], row[3], row[4], int(row[5]), row[6:]
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
            tweets[tid][GROUND_TRUTH_DIFF] = diff_label
    return tweets


def export_to_csv(dst_dir, tweet_dir, difficulty_path, anno_coll_name,
                  tweet_coll_name, cleaned, is_anonymous):
    """
    Creates csv datasets from the Mongo DB versions.

    Parameters
    ----------
    dst_dir: str - directory in which the csv files will be stored.
    tweet_dir: str - directory in which the original tweets are stored.
    difficulty_path: str - path to the csv file that contains the ground truth
    difficulty labels for all tweets.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.
    is_anonymous: bool - True if tweets shouldn't be included, False otherwise.

    """
    # Load ground truth difficulty labels
    diff_tweets = read_excel_tweets(difficulty_path)

    #########################################################################
    # Anonymizes the training set (500 tweets) as-is, i.e. using hierarchical
    # labeling scheme
    #########################################################################
    fname = "s0_tweets_hierarchical.csv"
    dst_tweets_s0_h = os.path.join(dst_dir, fname)
    fname = "s0_workers_hierarchical.csv"
    dst_workers_s0_h = os.path.join(dst_dir, fname)
    ds_name = "s0"
    is_flat = False

    tweets, workers = anonymize_train_hierarchical(
        ds_name, anno_coll_name=anno_coll_name,
        tweet_coll_name=tweet_coll_name, cleaned=cleaned)

    # Add ground truth difficulty labels
    for tid in tweets:
        tweets[tid][GROUND_TRUTH_DIFF] = diff_tweets[tid][GROUND_TRUTH_DIFF]

    store_tweets_csv(tweets, dst_tweets_s0_h, is_anonymous)
    store_workers_csv(workers, dst_workers_s0_h)

    # Test functions for reading the training datasets
    test1 = read_workers_from_csv(dst_workers_s0_h)
    print "#workers", len(test1)
    for w in test1:
        print w, test1[w]

    test2 = read_tweets_from_csv(dst_tweets_s0_h, is_flat)
    print "#tweets", len(test2)
    for w in test2:
        print w, test2[w]

    #########################################################################
    # Anonymizes the training set (500 tweets) s.t. it fits the crowdsourced
    # datasets, i.e. have only one label (flat labeling scheme): Irrelevant,
    # Neutral, Positive, or Negative according to majority voting
    #########################################################################
    fname = "s0_tweets_flat.csv"
    dst_tweets_s0_f = os.path.join(dst_dir, fname)
    fname = "s0_workers_flat.csv"
    dst_workers_s0_f = os.path.join(dst_dir, fname)
    is_flat = True
    ds_name = "S_0"

    tweets_s0, workers_s0 = anonymize_train_flat(
        ds_name, anno_coll_name=anno_coll_name,
        tweet_coll_name=tweet_coll_name, cleaned=cleaned)

    # Add ground truth difficulty labels
    for tid in tweets_s0:
        tweets_s0[tid][GROUND_TRUTH_DIFF] = diff_tweets[tid][GROUND_TRUTH_DIFF]

    store_tweets_csv(tweets_s0, dst_tweets_s0_f, is_anonymous)
    store_workers_csv(workers_s0, dst_workers_s0_f)

    # Test functions for reading the training datasets
    test1 = read_workers_from_csv(dst_workers_s0_f)
    print "#workers", len(test1)
    for w in test1:
        print w, test1[w]

    test2 = read_tweets_from_csv(dst_tweets_s0_f, is_flat)
    print "#tweetss", len(test2)
    for w in test2:
        print w, test2[w]

    #########################################################################
    # Anonymizes the crowdsourced sets (1k tweets each) s.t. only one label
    # (flat labeling scheme) is assigned per worker for a tweet: Irrelevant,
    # Neutral, Positive, or Negative according to majority voting
    #########################################################################
    # Crowdsourced dataset containing only predicted high disagreement tweets
    fname = "high_disagreement_tweets.csv"
    dst_tweets_h = os.path.join(dst_dir, fname)
    fname = "high_disagreement_workers.csv"
    dst_workers_h = os.path.join(dst_dir, fname)
    is_flat = True
    ds_name = "HIGH"
    # 8 votes per tweet
    labels = 8
    worker_offset = len(workers_s0)
    # Reason for calling it "LOW": we initially referred to tweets as tweets
    # with low agreement (which is the same as tweets with high disagreement)
    HIGH = os.path.join(base_dir, "results",
                       "dataset_twitter_crowdsourcing",
                       "Batch_2984090_batch_results_low_8000.csv")

    tweets_high, workers_high = anonymize_crowdsourced_dataset(
        HIGH, tweet_dir, labels, ds_name, worker_offset)

    # Add ground truth difficulty labels
    for tid in tweets_high:
        tweets_high[tid][GROUND_TRUTH_DIFF] = \
            diff_tweets[tid][GROUND_TRUTH_DIFF]

    store_tweets_csv(tweets_high, dst_tweets_h, is_anonymous)
    store_workers_csv(workers_high, dst_workers_h)

    # Test functions for reading the training datasets
    test1 = read_workers_from_csv(dst_workers_h)
    print "#workers", len(test1)
    for w in test1:
        print w, test1[w]

    test2 = read_tweets_from_csv(dst_tweets_h, is_flat)
    print "#tweetss", len(test2)
    for w in test2:
        print w, test2[w]

    # Crowdsourced dataset containing only predicted medium agreement tweets
    fname = "medium_disagreement_tweets.csv"
    dst_tweets_m = os.path.join(dst_dir, fname)
    fname = "medium_disagreement_workers.csv"
    dst_workers_m = os.path.join(dst_dir, fname)
    # 4 votes per tweet
    labels = 4
    is_flat = True
    ds_name = "MEDIUM"
    worker_offset = len(workers_s0) + len(workers_high)
    MEDIUM = os.path.join(base_dir, "results",
                          "dataset_twitter_crowdsourcing",
                          "Batch_2984078_batch_results_medium_4000.csv")
    tweets_medium, workers_medium = anonymize_crowdsourced_dataset(
        MEDIUM, tweet_dir, labels, ds_name, worker_offset)

    # Add ground truth difficulty labels
    for tid in tweets_medium:
        tweets_medium[tid][GROUND_TRUTH_DIFF] = \
            diff_tweets[tid][GROUND_TRUTH_DIFF]

    store_tweets_csv(tweets_medium, dst_tweets_m, is_anonymous)
    store_workers_csv(workers_medium, dst_workers_m)

    # Test functions for reading the training datasets
    test1 = read_workers_from_csv(dst_workers_m)
    print "#workers", len(test1)
    for w in test1:
        print w, test1[w]

    test2 = read_tweets_from_csv(dst_tweets_m, is_flat)
    print "#tweetss", len(test2)
    for w in test2:
        print w, test2[w]

    # Crowdsourced dataset containing only predicted low disagreement tweets
    fname = "low_disagreement_tweets.csv"
    dst_tweets_l = os.path.join(dst_dir, fname)
    fname = "low_disagreement_workers.csv"
    dst_workers_l = os.path.join(dst_dir, fname)
    # 4 votes per tweet
    labels = 4
    is_flat = True
    ds_name = "LOW"
    worker_offset = len(workers_s0) + len(workers_high) + len(workers_medium)
    LOW = os.path.join(base_dir, "results",
                        "dataset_twitter_crowdsourcing",
                        "Batch_2984071_batch_results_high_4000.csv")
    tweets_low, workers_low = anonymize_crowdsourced_dataset(
        LOW, tweet_dir, labels, ds_name, worker_offset)

    # Add ground truth difficulty labels
    for tid in tweets_low:
        tweets_low[tid][GROUND_TRUTH_DIFF] = diff_tweets[tid][GROUND_TRUTH_DIFF]

    store_tweets_csv(tweets_low, dst_tweets_l, is_anonymous)
    store_workers_csv(workers_low, dst_workers_l)

    # Test functions for reading the training datasets
    workers_low = read_workers_from_csv(dst_workers_l)
    print "#workers", len(workers_low)
    for w in workers_low:
        print w, workers_low[w]

    tweets_low = read_tweets_from_csv(dst_tweets_l, is_flat)
    print "#tweets", len(tweets_low)
    for w in tweets_low:
        print w, tweets_low[w]

    # Combine all datasets into a single file
    fname = "combined_tweets.csv"
    dst_tweets = os.path.join(dst_dir, fname)
    fname = "combined_workers.csv"
    dst_workers = os.path.join(dst_dir, fname)
    create_anonymized_combined_dataset(
        dst_tweets_s0_f, dst_workers_s0_f, dst_tweets_h, dst_workers_h,
        dst_tweets_m, dst_workers_m, dst_tweets_l, dst_workers_l, dst_tweets,
        dst_workers)

    # Test functions for reading the training datasets
    workers_high = read_workers_from_csv(dst_workers)
    print "#workers", len(workers_high)
    for w in workers_high:
        print w, workers_high[w]

    tweets_high = read_tweets_from_csv(dst_tweets, is_flat)
    print "#tweets", len(tweets_high)
    for w in tweets_high:
        print w, tweets_high[w]


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
            <GROUND_TRUTH_DIFF>: manual difficulty label,  # {A, E, I, B, O}
            <EXPLANATION>: explanation of <DIFFICULTY>
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
            tid, _, _, _, man_diff, explain = \
                row[0], row[1], row[2], row[3], row[4], row[5]
            # print tid, def_diff, man_diff
            # Some labels have whitespaces
            man_diff = man_diff.strip()
            # Sanity check - label exists for each tweet
            assert(len(man_diff) > 0)
            # Store info for tweet
            tweets[tid] = {}
            tweets[tid][EXPLANATION] = explain
            tweets[tid][GROUND_TRUTH_DIFF] = man_diff
            uniq.add(man_diff)
    return tweets


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    cleaned = True
    # Name of the collection in each DB holding annotator data
    ANNO_COLL_NAME = "user"
    # Name of the collection in each DB holding tweet data
    TWEET_COLL_NAME = "tweets"

    # Directory in which all acceptable tweets without URLs are stored
    TWEETS = "/media/data/dataset/debate1_trump_vs_clinton_sanitized/"

    # Read in manually labeled (w.r.t. difficulty) tweets that were labeled
    # using Excel
    src = os.path.join(base_dir, "results", "ambiguous_difficult",
                       "difficulty_labels_all_votes_correct_labeled.csv")

    # 1. Export tweets anonymized
    # Directory in which datasets will be stored
    DS_DIR = os.path.join(base_dir, "results", "anonymous")
    if not os.path.exists(DS_DIR):
        os.makedirs(DS_DIR)
    is_anonymous = True
    export_to_csv(DS_DIR, TWEETS, src, ANNO_COLL_NAME, TWEET_COLL_NAME, cleaned,
                  is_anonymous)

    # 2. Export tweets with texts
    DS_DIR = os.path.join(base_dir, "results", "export")
    if not os.path.exists(DS_DIR):
        os.makedirs(DS_DIR)
    is_anonymous = False
    export_to_csv(DS_DIR, TWEETS, src, ANNO_COLL_NAME, TWEET_COLL_NAME, cleaned,
                  is_anonymous)

    # tweets_high = read_tweets_from_csv(os.path.join(base_dir, "results", "export", ""), is_flat)
    # print "#tweets", len(tweets_high)
    # for w in tweets_high:
    #     print w, tweets_high[w]