"""
For some tweets of the Twitter dataset there are no Watson and/or Rosette
features or some other features from Gizem might be missing. Thus, we only keep
valid ones.

"""
import os
import codecs
import json
import unicodecsv as csv
from collections import Counter
import random
import copy

import numpy as np

import utility
from anno import Annotator


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

# Names and order in which the attributes occur in the file
# Same as in <add_tweet_ids_and_expert_labels_to_gizems_crowdsourced_dataset.py
# I also copy&pasted the features from there
FEATURE_NAMES = ['tweet_AvgPol_Twitter', 'tweet_AvgPol_TwitterRatio',
                 'tweet_minPolToken_Twitter', 'tweet_maxPolToken_Twitter',
                 'tweetFirst_AvgPol_Twitter', 'tweetFirst_AvgPol_TwitterRatio',
                 'tweetFirst_minPolToken_Twitter',
                 'tweetFirst_maxPolToken_Twitter', 'tweetSecond_AvgPol_Twitter',
                 'tweetSecond_AvgPol_TwitterRatio',
                 'tweetSecond_minPolToken_Twitter',
                 'tweetSecond_maxPolToken_Twitter', 'tweet_AvgPol_Senti',
                 'tweet_AvgPol_SentiRatio', 'tweet_DomPol_Senti',
                 'tweet_DomPol_SentiRatio', 'tweet_minPolToken_Senti',
                 'tweet_maxPolToken_Senti', 'tweetFirst_AvgPol_Senti',
                 'tweetFirst_AvgPol_SentiRatio', 'tweetFirst_DomPol_Senti',
                 'tweetFirst_DomPol_SentiRatio', 'tweetSecond_AvgPol_Senti',
                 'tweetSecond_AvgPol_SentiRatio', 'tweetSecond_DomPol_Senti',
                 'tweetSecond_DomPol_SentiRatio', 'tweet_textBlobSentiment',
                 'tweet_textBlobSentimentRatio', 'tweetFirst_textBlobSentiment',
                 'tweetFirst_textBlobSentimentRatio',
                 'tweetSecond_textBlobSentiment',
                 'tweetSecond_textBlobSentimentRatio', 'tweetLength',
                 'beingRetweetOrNot', 'tweet_PosWords', 'tweet_NegWords',
                 'tweet_PosTermRatio', 'tweet_NegTermRatio', 'tweetFirstLength',
                 'tweetSecondLength', 'tweetFirst_PosWords',
                 'tweetFirst_PosTermRatio', 'tweetFirst_NegWords',
                 'tweetFirst_NegTermRatio', 'tweetSecond_PosWords',
                 'tweetSecond_PosTermRatio', 'tweetSecond_NegWords',
                 'tweetSecond_NegTermRatio', 'tweet_SumFreq', 'tweet_MeanFreq',
                 'tweet_MinFreq', 'tweet_MaxFreq', 'tweet_VarianceFreq',
                 'tweet_NounsNum', 'tweet_AdjsNum', 'tweet_AdvsNum',
                 'tweet_VerbsNum', 'tweet_NounPercentage',
                 'tweet_AdjPercentage', 'tweet_AdvPercentage',
                 'tweet_VerbPercentage', 'numQueryOccurrenceInTweet',
                 'tweet_numExcMarks', 'tweet_numQuestMarks',
                 'tweet_numSuspensionPoints', 'tweet_numQuotationMarks',
                 'tweet_numKeywordWould', 'tweet_numKeywordLike',
                 'tweet_numKeywordSudden', 'tweet_numKeywordYet',
                 'tweet_numTwitterLingos', 'tweet_numPosEmots',
                 'tweet_numNegEmots', 'tweet_numExcMarksRatio',
                 'tweet_numQuestMarksRatio', 'tweet_numPosEmotsRatio',
                 'tweet_numNegEmotsRatio', 'tweet_numAllUppercaseTokens',
                 'tweet_numAllUppercaseTokensRatio',
                 'tweet_numRepeatingCharactersTokens',
                 'tweet_numRepeatingCharactersTokensRatio',
                 'levenshteinDistance', 'jaccardSimilarityShingle1',
                 'jaccardSimilarityShingle2', 'tweet_probabilityLDATopic1',
                 'tweet_probabilityLDATopic2', 'tweet_probabilityLDATopic3',
                 'tweet_probabilityLDATopic4', 'tweet_probabilityLDATopic5',
                 'tweet_probabilityLDATopic6', 'tweet_probabilityLDATopic7',
                 'tweet_probabilityLDATopic8', 'tweet_probabilityLDATopic9',
                 'tweet_probabilityLDATopic10']

SEED = 13
random.seed(SEED)


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
        # Extract data only if all information is available, else skip tweet
        if "keywords" in data and "sentiment" in data and "entities" in data:
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
                block = (dic["text"].lower(), disambiguation,
                         entity_type.lower())
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
        if "POS" in data and "SENTIMENT" in data:
            # a) POS tags
            pos_tags = []
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


def read_gizem_features(src_features, src_header):
    """
    Extracts Gizem's computed features.

    Parameters
    ----------
    src_features: str - path to arff file with features.
    src_header: str - path to file containing feature names of <src_features>.

    Returns
    -------
    dict.
    {
        tid:
        {
            feature1: ...,
            feature2: ---,
        }
    }


    """
    features = []
    # 1. Read feature names
    with open(src_header, "rb") as f:
        lines = f.readlines()
    # Skip header
    for line in lines[2:]:
        if line.startswith("@ATTRIBUTE"):
            features.append(line.split(" ")[1])
    # Ignore class label
    features = features[:-1]
    # DON'T DELETE THIS OUTPUT because we can copy&paste it into other files
    print "FEATURE NAMES"
    print features

    # tid:
    # {
    #     feature1: ...,    # feature1 is the extracted name from the arff file
    #     feature2: ---,
    # }
    data = {}
    with open(src_features, "rb") as f:
        feature_lines = f.readlines()
    for fline in feature_lines:
        tid, rest = fline.split("\t")
        values = rest.split(",")
        # Get rid of \n\r
        values[-1] = values[-1].strip()
        data[tid] = {}
        # Ignore LDA features, because for 300 tweets they are missing
        # No idea if it's a bug or not, but to be safe, let's ignore all LDA
        # features!
        for fname, value in zip(features, values):
            # if not fname.startswith("tweet_probabilityLDATopic"):
            data[tid][fname] = value
    print "#tweets", len(data)
    print "#features", len(data[data.keys()[0]])
    return data


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
            "username": username,
            "query": topic
        }
    }
    """
    with codecs.open(src, "r", encoding="utf-8") as f:
        tweets = json.load(f, encoding="utf-8")
    return tweets


def read_twitter_csv(src):
    """
    Reads Twitter dataset in .csv format

    Parameters
    ----------
    src: str - path to csv file.

    Returns
    -------
    dict.
    {tid:
            "text": ...
    }

    """
    data = {}
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            tid, text = row
            data[tid] = text
    return data


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


def create_dataset(watson_dir, rosette_dir, anno_coll_name="user",
        tweet_coll_name="tweets", cleaned=False, min_annos=3):
    """
    Creates a dataset comprising all tweets (regardless of institution and
    annotator group) that were labeled at least <min_annos> times. Dataset is
    stored as csv with "," as a separator and a header in the first line.

    Parameters
    ----------
    institution name or group and the thresholds are the corresponding values.
    anno_coll_name: str - name of the collection holding the annotator data.
    tweet_coll_name: str - name of the collection holding the tweet data.
    cleaned: bool - True if the cleaned data is used as input.
    min_annos: int - minimum number of annotators who must've assigned a label
    to a tweet. Otherwise it'll be discarded.

    Returns
    -------
    dict.
    {
        tid:
            {
            "...": "..."
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
    annos, counts = read_dataset(
        DB_NAMES, dbs, anno_coll_name=anno_coll_name,
        tweet_coll_name=tweet_coll_name, cleaned=cleaned, min_annos=min_annos)
    # Store tweets in dict:
    # {tid: text, number of annos who labeled it, 3 labels anno1,
    # 3 labels anno 2,
    # 3 labels anno3 etc., disagreement label
    #
    # }
    data = aggregate_data_per_tweet(annos, counts)

    # Uncomment to export tweets for Gizem
    # export_dst = "/media/data/Workspaces/PythonWorkspace/phd/Analyze-Labeled-Dataset/www2018_results/dataset_twitter/tweets_gizem_training.csv"
    # with open(export_dst, "wb") as f:
    #     writer = csv.writer(f, dialect='excel', encoding='utf-8',
    #                         delimiter="\t")
    #     for tid in data:
    #         text = data[tid]["text"]
    #         # Remove all line breaks in a tweet
    #         text = text.replace('\n', ' ').replace('\r', '')
    #         writer.writerow([tid, text])

    watson = get_watson_data(watson_dir)
    rosette = get_rosette_data(rosette_dir)
    # Merge dictionaries - all have the same keys
    merged = merge_dicts(data, watson, rosette)
    return merged


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Directory where the merged dataset will be stored
    DST_DIR = os.path.join(base_dir, "results", "dataset_twitter")
    # Path to Gizem's extracted features
    FEATURES = os.path.join(base_dir, "results", "dataset_twitter",
                            "FeatureFile_Trump_Labelled_500.txt")
    # Path to header file containing the feature names and their order
    HEADER = os.path.join(base_dir, "results", "dataset_twitter",
                          "Features_Trump.txt")
    # Path to Rosette features
    ROSETTE_DIR = os.path.join(base_dir, "results", "rosette_sentiment_twitter")
    # Path to Watson features
    WATSON_DIR = os.path.join(base_dir, "results", "watson_sentiment_twitter")

    cleaned = True
    min_annos = 3
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    # Name of the collection in each DB holding annotator data
    ANNO_COLL_NAME = "user"
    # Name of the collection in each DB holding tweet data
    TWEET_COLL_NAME = "tweets"
    dataset = create_dataset(WATSON_DIR, ROSETTE_DIR,
                             anno_coll_name=ANNO_COLL_NAME,
                             tweet_coll_name=TWEET_COLL_NAME, cleaned=cleaned,
                             min_annos=min_annos)

    fname = "dataset_twitter_watson_rosette_{}.json".format(agg)
    DST = os.path.join(DST_DIR, fname)

    # {tid: {f1: .., f2:..}
    features = read_gizem_features(FEATURES, HEADER)

    # {tid: {...}}
    watson = get_watson_data(WATSON_DIR)
    # {tid: {...}}
    rosette = get_rosette_data(ROSETTE_DIR)

    # {tid:
    #     {
    #         "text": ...,
    #     }
    # }
    if not os.path.isdir(ROSETTE_DIR):
        raise IOError("Rosette sentiment directory doesn't exist - run '"
                      "extract_sentiment_ner_rosette.py' first!")
    if not os.path.isdir(WATSON_DIR):
        raise IOError("Watson sentiment directory doesn't exist - run '"
                      "extract_sentiment_ner_watson.py' first!")

    # Store updated dataset with expert labels and only consider tweets for
    # which we have Rosette and Watson features
    # {tid:
    #      {
    #          "feature1": "...", # all features from MongoDB and more stuff
    #      }
    # }
    tweets = {}
    with codecs.open(DST, "w", encoding="utf-8") as f:
        for tid in dataset:
            # Only consider tweets for which we have all features
            if tid in watson and tid in rosette and tid in features:
                text = dataset[tid]["text"]
                # Remove all line breaks in a tweet
                text = text.replace('\n', ' ').replace('\r', '')
                dataset[tid]["text"] = text
                tweets[tid] = dataset[tid]
                # Add Gizem's features
                tweets[tid].update(features[tid])
        # https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
        data = json.dumps(tweets, encoding="utf-8", indent=4,
                          ensure_ascii=False)
        f.writelines(unicode(data))
    data = read_gizem_json(DST)
    print "#tweets", len(data)
