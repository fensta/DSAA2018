"""
Creates the dataset used for this paper. We use all tweets (regardless of
institution and group) that were labeled at least N times. We create a
dataset for predicting the agreement of annotators. Adds labels for agreement
prediction task as well as sentiment analysis task.

"""
import os
from collections import Counter
import codecs
import json
import copy
import math
import random
import unicodecsv as csv

import matplotlib.pyplot as plt
import numpy as np

from anno import Annotator
import utility


# Default label assigned by annotator
EMPTY = ""
# Default value for annotation time
ZERO = 0

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
    "Negative": "Negative"
}

SEED = 13
random.seed(SEED)


def create_dataset_full(src_full, dst, watson_dir,
                        rosette_dir):
    """
    Creates a dataset comprising all tweets (regardless of institution and
    annotator group) that were labeled at least <min_annos> times. Dataset is
    stored as json.

    Parameters
    ----------
    src_full: str - path to full TREC dataset.
    dst: str - path where dataset will be stored.
    watson_dir: str - path where the features extracted via Watson are stored.
    rosette_dir: str - path where the features extracted via Rosette are stored.

    """
    tweets = read_gizem_json(src_full)
    watson = get_watson_data(watson_dir)
    rosette = get_rosette_data(rosette_dir)
    merged = merge_dicts(tweets, watson, rosette)
    store_json(merged, dst)


def read_gizem_json(src):
    """
    Reads in Gizem's crowdsourced dataset in json format that contains
    expert and crowd labels.

    Parameters
    ----------
    src: str - path to Gizem's crowdsourced TREC dataset.

    Returns
    -------
    dict.

    {tid:
        {
            "text": text,
            "expert_label": expert_label,
            "crowd_labels": crowd_labels,
            "username": username,
            "query": topic
        }
    }
    """
    with codecs.open(src, "r", encoding="utf-8") as f:
        tweets = json.load(f, encoding="utf-8")
    return tweets


def merge_dicts(tweets, watson, rosette):
    """
    Merges a list of dictionaries. If a value doesn't exist in either <rosette>
    or <watson>, the tweet is discarded (because the tweet was written in a
    non-English language and hence not all features would be available for it).
    The first dictionary represents the tweets, the other two dictionaries the
    Watson and Rosette dictionaries that were created using their APIs.
    The first dictionary is updated with the values of the remaining ones.
    Keys in the dictionaries are always the tweet IDs.

    Params
    ------
    dicts: dict - represents the tweets.
    watson: dict - represents features extracted for tweets by Watson API.
    rosette: dict - represents features extracted for tweets by Rosette API.

    Returns
    -------
    dict.
    Merged dictionary. Contains only the tweets for which <watson> and <rosette>
    entries existed.

    """
    cnt = 0
    res = copy.deepcopy(tweets)
    # For each key
    for tid in tweets:
        if tid in watson and tid in rosette:
            # Go through remaining dictionaries
            # Append each key of both dictionaries
            for k in watson[tid]:
                res[tid][k] = watson[tid][k]
            for k in rosette[tid]:
                res[tid][k] = rosette[tid][k]
        else:
            # print "tweet deleted: {}".format(tid)
            del res[tid]
            cnt += 1
    print "#deleted", cnt
    return res


def read_dataset(
        dbs, db_idxs, anno_coll_name="user",
        tweet_coll_name="tweets", cleaned=False, min_annos=3):
    """
    Read dataset.

    Parameters
    ----------
    dbs: list of strings - names of the existing DBs.
    db_idxs: list of ints - name of the MongoDB from where data should be read.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.
    is_early: bool - True if only tweets from the early phase should be
    considered. Else only tweets from the late stage are considered.

    Returns
    --------
    list, dict.
    List of Annotator objects in institution with their tweets.
    Dictionary storing for each tweet how many annotators labeled it.

    """
    # Store a list of annotators per group/institution
    inst_annos = []

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
            inst_anno = Annotator(dict_key, group)
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
                anno_time = sum([rel_time, fac_time, opi_time])
                labels = [first_label, second_label, third_label]
                inst_anno.add_tweet(tweet_id, anno_time, labels, text)
            # Store annotator
            inst_annos.append(inst_anno)

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


def compute_medians(annos):
    """
    Computes the median labeling costs per tweet.

    Parameters
    ----------
    annos: list of anno.Annotator - annotators.

    Returns
    -------
    dict.
    Tweet IDs as keys and median labeling costs of a tweet as a value.

    """
    # Get annotation times per tweet
    tweets = {
        # {tweet_id: [costs for anno1, costs for anno2...]}
    }
    for anno in annos:
        for idx, tid in enumerate(anno.tweets.ids):
            if tid not in tweets:
                tweets[tid] = []
            # Add costs to group and institution
            tweets[tid].append(anno.tweets.anno_times[idx])
    median_costs = {}
    for tid in tweets:
        median_costs[tid] = np.median(np.array(tweets[tid]))
    return median_costs


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
    {tid: text, number of annos who labeled it, 3 labels anno1,
    3 labels anno 2,
    3 labels anno3 etc., disagreement score, labeling cost

    }

    """
    costs = compute_medians(annos)
    tweets = {}
    for anno in annos:
        for idx, tid in enumerate(anno.all_tweets()):
            # New tweet that was sufficiently often labeled
            if tid in counts and tid not in tweets:
                tweets[tid] = {
                    "text": anno.tweets.texts[idx],
                    "votes": counts[tid],
                    # List of lists: inner list represents all labels assigned
                    # by a single annotator
                    "labels": [],
                    "agree_score": 0.0,
                    "labeling_cost": costs[tid]
                }
            tweets[tid]["labels"].append(anno.tweets.labels[idx])
    # Compute agreement score
    for tid in tweets:
        tweets[tid]["agree_score"] = \
            compute_agreement_score(tweets[tid]["labels"])
    # Derive ground truth
    for tid in tweets:
        relevance_label, sentiment_label = \
            derive_ground_truth(tweets[tid]["labels"])
        tweets[tid]["relevance_label"] = relevance_label
        tweets[tid]["sentiment_label"] = sentiment_label
    return tweets


def derive_ground_truth(labels):
    """
    Derives the ground truth from the dataset using majority voting.
    a) Relevant or Irrelevant and if Relevant, then also Positive or
    Negative or Neutral.
    However, the function returns regardless of relevance the majority label
    that isn't empty (indicating that no label was assigned). This is due
    to the sentiment analysis task because we first predict relevance.
    If an irrelevant tweet is predicted as relevant, we need a true sentiment
    label instead of an empty (although correct) label. If no other label than
    "Irrelevant" was assigned, a random sentiment label is returned.

    Parameters
    ----------
    labels: list of lists: inner list represents all labels assigned
    by a single annotator

    Returns
    -------
    str, str.
    Relevance and sentiment label. The latter might be empty indicating that
    most chose "Irrelevant" for a tweet.

    """
    sent_label = ""
    # Group labels according to level
    first_level, second_level, third_level = zip(*labels)
    # Count label occurrences per level
    first = Counter(first_level)
    second = Counter(second_level)
    third = Counter(third_level)
    rel_label, _ = first.most_common(1)[0]
    rel_label = GROUND_TRUTH[rel_label]
    # CORRECT IMPLEMENTATION: return majority label that isn't empty if tweet
    # is relevant, else return emtpy
    # if rel_label == "Relevant":
    #     # Select the most-selected non-empty label
    #     # This scenario occurs for example if:
    #     # on first level: [(u'Relevant', 15), (u'Irrelevant', 12)]
    #     # on second level: [('', 12), (u'Non-factual', 9), (u'Factual', 6)]
    #     # Since it's relevant, it needs a non-empty label
    #     for label, count in second.most_common():
    #         # Found the label
    #         if len(label) > 0:
    #             sent_label = label
    #             break
    #     sent_label = GROUND_TRUTH[sent_label]
    #     if sent_label != "Neutral":
    #         #  Select the most-selected non-empty label
    #         for label, count in third.most_common():
    #             # Found the label
    #             if len(label) > 0:
    #                 sent_label = label
    #                 break
    #         sent_label = GROUND_TRUTH[sent_label]
    # Return majority non-empty label regardless of relevance -> necessary
    # for the sentiment analysis task
    # Select the most-selected non-empty label
    # This scenario occurs for example if:
    # on first level: [(u'Relevant', 15), (u'Irrelevant', 12)]
    # on second level: [('', 12), (u'Non-factual', 9), (u'Factual', 6)]
    # Since it's relevant, it needs a non-empty label
    for label, count in second.most_common():
        # Found the label
        if len(label) > 0:
            sent_label = label
            break
    # TODO: is this implementation ok or should we change anything? Perhaps
    # use uncleaned dataset to figure out which sentiment is prevalent
    # If tweet was assigned no other label other than "irrelevant", assign one
    # randomly
    if len(sent_label) == 0:
        sent_label = random.choice(GROUND_TRUTH.values())
        print "randomly assigned sentiment label", sent_label
    else:
        sent_label = GROUND_TRUTH[sent_label]
        if sent_label != "Neutral":
            #  Select the most-selected non-empty label
            for label, count in third.most_common():
                # Found the label
                if len(label) > 0:
                    sent_label = label
                    break
            sent_label = GROUND_TRUTH[sent_label]
    return rel_label, sent_label


def compute_agreement_score(labels):
    """
    Computes the annotator agreement for a tweet given all its labels for the
    different hierarchical levels.

    Parameters
    ----------
    labels: list of list: inner list represents all labels assigned
    by a single annotator

    Returns
    -------
    float.
    Agreement score between 0-1, 1 being high disagreement.

    """
    # Group labels according to level
    first_level, second_level, third_level = zip(*labels)
    # Count label occurrences per level
    first = Counter(first_level)
    second = Counter(second_level)
    third = Counter(third_level)
    # Compute agreement score
    score, _ = agreement([first, second, third])
    return score


def agreement(tweet):
    """
    Computes the agreement for a given tweet. The agreement of each hierarchy
    level contributes to the final agreement according to the number of people
    who agree on the majority label.

    That implies that the higher levels in the hierarchy contribute more to
    agreement (it's also easier to assign those).

    Weighted agreement formula:
    Agreement = agree1/total_agree * agree1/level_labels1 +
    agree2/total_agree * agree2/level_labels2 + agree3/total_agree *
    agree3/level_labels3

    - agreeX: number of annotators on level X that agree on the label
    - total_agree: sum of all annotators across all levels that agree on the
    labels
    - level_labelsX: number of annotators who assigned a label on level X
    NOTE THAT in a tweet the levels with mixed labels (= no majority) decrease
    the influence of levels having a majority (for each such level, total_agree
    is increased by 1)


    Parameters
    ----------
    tweet: list of Counter - each Counter holds the counts for the labels of
    that level in the hierarchy.

    Returns
    -------
    float, int.
    Level of average agreement for the given tweet, number of annotators who
    labeled the tweet

    """
    # Count of all annotators that agree on majority labels over all levels
    agree_total = 0
    # Total votes of annotators on each hierarchy level
    level_labels = []
    # Number of annotators agreeing on the label per hierarchy level
    agrees = []
    annos = 0
    for idx, level in enumerate(tweet):
        # print "labels:", level
        votes_per_level = sum(level.values())
        # Count annotators who labeled a tweet
        if idx == 0:
            annos = votes_per_level
        level_labels.append(votes_per_level)
        # Number of annotators agreeing on the label
        votes = max(level.values())
        agree = 1.0 * votes
        agree_total += agree
        # If a level has mixed labels (= no majority label), penalize it by
        # increasing the contribution of such levels
        if abs(agree / votes_per_level - 0.5) < 0.001:
            agree = 1.0
            agree_total += 1
            # print "found a mixed level - penalize tweet"
        agrees.append(agree)

    # Weight by which the agreement of each level is weighed
    agree_weights = []
    for votes in agrees:
        weight = 0
        if agree_total > 0:
            weight = 1.0 * votes / agree_total
        agree_weights.append(weight)
        # print "weight", weight
    agreement = 0.0
    for idx, (weight, agree) in enumerate(zip(agree_weights, agrees)):
        # print weight, "*", agree, "/", level_labels[idx]
        agreement += weight * 1.0 * agree / level_labels[idx]
    return agreement, annos


def store_json(tweets, dst):
    """
    Stores dataset in json format. Stores missing labels as "?".

    Parameters
    ----------
    tweets: dict -  For each tweet ID the data to be written to file (removes
    "labels" and also any new lines from "text")
    dst: str - path where output will be stored.

    """
    with codecs.open(dst, "w", encoding="utf-8") as f:
        for tid in tweets:
            t = tweets[tid]
            text = t["text"]
            # Remove all line breaks in a tweet
            text = text.replace('\n', ' ').replace('\r', '')
            t["text"] = text

            # Flatten labels
            # flattened = [label for anno_labels in t["labels"] for label
            #              in anno_labels]
            # preserved = []
            # # Preserve '' indicating that the label is missing
            # for label in flattened:
            #     if len(label) == 0:
            #         label = "?"
            #     preserved.append(label)
            if "labels" in t:
                del t["labels"]
            # t["labels"] = preserved
            # preserved = ",".join(preserved)
        # https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
        data = json.dumps(tweets, encoding="utf-8", indent=4,
                          ensure_ascii=False)
        f.writelines(unicode(data))


def get_watson_data(src_dir):
    """
    Extracts overall tweet sentiment, NERs (lowercase), and keywords
    (lowercase). Sentiment per keyword as well as sentiment per NER is ignored,
    because they're mostly 0, so it's pointless.

    Parameters
    ----------
    src_dir: str - directory in which the sentiment data is stored.

    Returns
    -------
    dict.
    { tid: {
                "keywords": [],
                "watson_sentiment": [],
                "watson_ners": []
           }
    }

    """
    tweets = {}
    # For each file
    for fname in os.listdir(src_dir):
        tid = fname.split(".")[0]
        fpath = os.path.join(src_dir, fname)
        with codecs.open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f, encoding="utf-8")
        # Extract data
        # a) Keywords (lowercase)
        keywords = []
        for dic in data["keywords"]:
            keywords.append(dic["text"].lower())
        # b) Overall sentiment
        sentiment = data["sentiment"]["document"]["score"]
        # c) NERs (lowercase)
        entities = []
        for dic in data["entities"]:
            entity_type = dic["type"]
            disambiguation = None
            if "disambiguation" in dic:
                # Lowercase
                disambiguation = [term.lower() for term in
                                  dic["disambiguation"]["subtype"]]
            # Store (text in tweet, [disambiguation terms], type of NER)
            block = (dic["text"].lower(), disambiguation, entity_type.lower())
            entities.append(block)
        tweets[tid] = {
            "keywords": keywords,
            "watson_sentiment": sentiment,
            "watson_ners": entities
        }
    return tweets


def get_rosette_data(src_dir):
    """
    Extracts POS-tags (without punctuation) and NERs (lowercase).
    Sentiment per NER and overall
    sentiment are ignored as they only provide labels, but no scores, plus
    NER sentiment is mainly neutral, i.e. insufficient context.

    Parameters
    ----------
    src_dir: str - directory in which the sentiment data is stored.

    Returns
    -------
    dict.
    { tid: {
                "pos_tags": [],
                "rosette_ners": []
           }
    }

    """
    tweets = {}
    # For each file
    for fname in os.listdir(src_dir):
        tid = fname.split(".")[0]
        fpath = os.path.join(src_dir, fname)
        with codecs.open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f, encoding="utf-8")
        # Extract data
        # a) POS tags
        pos_tags = []
        if "POS" in data and "SENTIMENT" in data:
            # Remove POS-tags for punctuation because we're not interested in it
            tags = data["POS"]["posTags"]
            for tag in tags:
                if tag != "PUNCT":
                    pos_tags.append(tag)
            # b) NERs (lowercase)
            entities = []
            for entity in data["SENTIMENT"]["entities"]:
                # Store (text in tweet, resolved NER, type of NER)
                block = (entity["mention"].lower(), entity["normalized"].lower(),
                         entity["type"].lower())
                entities.append(block)
            tweets[tid] = {
                "pos_tags": pos_tags,
                "rosette_ners": entities
            }
    return tweets


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


def read_json(src):
    """
    Reads in a json file.

    Parameters
    ----------
    src: str - path to input json file.

    Returns
    -------
    dict.
    For each tweet ID the data to be written to file:
    {tid: text, number of annos who labeled it, 3 labels anno1,
    3 labels anno 2,
    3 labels anno3 etc., disagreement score
    }

    """
    # https://stackoverflow.com/questions/23933784/reading-utf-8-escape-sequences-from-a-file
    # https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
    with codecs.open(src, "r", encoding="utf-8") as f:
        tweets = json.load(f, encoding="utf-8")
    for t in tweets:
        # Remove all line breaks in a tweet
        # Replace all "&" by "and
        # Replace all "\'" (or \u2019 or \u2018 or \u0027 or \u2032) by "'"
        # Replace all "..." (or \u2026) by " "
        # Replace all ".." (or \u2025) by " "
        # Replace all "&gt;" with "larger than"
        # Replace all "&lt;" with "smaller than"
        # Replace all "%" with " percent"
        # Replace all "-" with " "
        # Replace all "=" with " equals "
        # Replace all "1st" with " first "
        # Replace all "2nd" with " second "
        # Replace all "3rd" with " third "
        # Replace all "y/o" with " years old "
        # Replace all "ur" with " your "
        # Replace all " r " with " you are "
        # Replace all " yr " with " year "
        # Replace all ". . . ." with " "
        # Replace all ". . ." with " "
        # Replace all "ppl" with "people
        # Convert to UTF-8 format
        text = tweets[t]["text"]
        text = text.replace('\n', ' ').replace('\r', '')\
            .replace("\u2026", " ").replace("...", " ")\
            .replace("..", " ").replace("\u2025", " ")\
            .replace("\u2032", "'").replace("\u2019", "'")\
            .replace("\u0027", "'").replace("\u2018", "'") \
            .replace("&amp;", "and") \
            .replace("\\", "") \
            .replace("&gt;", "larger than") \
            .replace("&lt;", "smaller than") \
            .replace("%", " percent") \
            .replace("-", " ") \
            .replace("=", " equals ") \
            .replace("1st", " first ") \
            .replace("2nd", " second ") \
            .replace("3rd", " third ") \
            .replace("y/o", " years old ") \
            .replace(" ur ", "your") \
            .replace(" r ", " you are ") \
            .replace(" yr ", " year ") \
            .replace(". . . .", " ") \
            .replace(". . .", " ") \
            .replace("ppl", "people")

        # Remove all double (or more) whitespaces
        text = " ".join(text.split())
        # Overwrite existing text
        tweets[t]["text"] = text

    return tweets


def plot_agreement_scores(tweets, dst):
    """
    Plots the actual agreement scores in the dataset.

    Parameters
    -----------
    tweets: dict - tweets, see read_csv().
    dst: str - path where plot will be stored.

    """
    x = range(len(tweets))
    y = []
    for tid in tweets:
        y.append(tweets[tid]["agree_score"])
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # Sort y-values ascendingly
    y = sorted(y)
    ax.scatter(x, y)
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Limits of axes
    x = ax.get_xlim()
    plt.xlim(0, x[1])
    plt.ylim(0, 1.1)
    # Set labels of axes
    ax.set_xlabel("Tweet ID")
    ax.set_ylabel("Difficulty score")
    plt.savefig(dst, bbox_inches='tight')
    plt.close()


def plot_agreement_score_distribution(tweets, dst):
    """
    Plots the actual agreement scores in the dataset.

    Parameters
    -----------
    tweets: dict - tweets, see read_csv().
    dst: str - path where plot will be stored.

    """
    x = range(len(tweets))
    y = []
    for tid in tweets:
        y.append(tweets[tid]["agree_score"])
    fig = plt.figure(figsize=(5, 3))

    bins = int(round(math.sqrt(len(tweets))))
    print "BINS:", bins
    fig = plt.figure(figsize=(5, 3))
    ax = fig.add_subplot(111)
    # x = range(max(anno_times))

    # Make sure that the sum of the bars equals 1
    # https://stackoverflow.com/questions/3866520/plotting-histograms-whose-bar-heights-sum-to-1-in-matplotlib/16399202#16399202
    weights = np.ones_like(y) / float(len(y))
    n, bins, _ = ax.hist(y, bins=bins, weights=weights,
                         histtype='stepfilled', alpha=0.2)
    ax = fig.add_subplot(111)
    # Sort y-values ascendingly
    # y = sorted(y)
    # ax.scatter(x, y)
    # Hide the right and top spines (lines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    # Limits of axes
    x = ax.get_xlim()
    # plt.xlim(0, x[1])
    # plt.ylim(0, 1.1)
    # Set labels of axes
    ax.set_xlabel("Agreement score")
    ax.set_ylabel("Probability")
    plt.savefig(dst, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Directory in which the figures will be stored
    FIG_DIR = os.path.join(base_dir, "results", "figures")
    # Directory in which dataset will be stored
    DS_DIR = os.path.join(base_dir, "results", "dataset_twitter")
    ROSETTE_DIR = os.path.join(base_dir, "results", "rosette_sentiment_twitter")
    WATSON_DIR = os.path.join(base_dir, "results", "watson_sentiment_twitter")
    # Directory in which the full TREC dataset is stored
    FULL_DIR = os.path.join(base_dir, "results", "dataset_twitter_full")
    if not os.path.exists(DS_DIR):
        os.makedirs(DS_DIR)
    if not os.path.exists(FIG_DIR):
        os.makedirs(FIG_DIR)

    if not os.path.isdir(ROSETTE_DIR):
        raise IOError("Rosette sentiment directory doesn't exist - run '"
                      "extract_sentiment_ner_rosette.py' first!")
    if not os.path.isdir(WATSON_DIR):
        raise IOError("Watson sentiment directory doesn't exist - run '"
                      "extract_sentiment_ner_watson.py' first!")
    cleaned = True
    min_annos = 3
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    # Name of the collection in each DB holding annotator data
    ANNO_COLL_NAME = "user"
    # Name of the collection in each DB holding tweet data
    TWEET_COLL_NAME = "tweets"
    fname = "dataset_twitter_watson_rosette_{}.json".format(agg)
    dst = os.path.join(DS_DIR, fname)

    ########################
    # Create training set  #
    ########################
    # Was already created in add_gizems_features_to_twitter_dataset.py

    # Read in dataset for testing
    dataset = read_json(dst)
    for tid in dataset:
        print dataset[tid]

    # Plot agreement scores
    fname = "dataset_twitter_min_annos_{}_agreement_scores_{}.pdf"\
        .format(min_annos, agg)
    dst = os.path.join(FIG_DIR, fname)
    plot_agreement_scores(dataset, dst)

    # # Plot distribution of agreement scores
    fname = "dataset_twitter_min_annos_{}_agreement_score_distribution_{}.pdf"\
        .format(min_annos, agg)
    dst = os.path.join(FIG_DIR, fname)
    plot_agreement_score_distribution(dataset, dst)

    ###################
    # Create test set #
    ###################
    # But ignore all tweets that were labeled in crowdsourced TREC dataset
    ROSETTE_DIR = os.path.join(base_dir, "results",
                               "rosette_sentiment_twitter_full")
    WATSON_DIR = os.path.join(base_dir, "results",
                              "watson_sentiment_twitter_full")
    fname = "dataset_twitter_full_watson_rosette_{}.json".format(agg)
    dst = os.path.join(FULL_DIR, fname)
    src_full = os.path.join(FULL_DIR, "twitter_full_gizem_features.json")
    create_dataset_full(src_full, dst, WATSON_DIR, ROSETTE_DIR)

    # Read in dataset for testing
    dataset = read_json(dst)
    for tid in dataset:
        print dataset[tid]
