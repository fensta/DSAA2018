"""
Evaluates the predictions from Weka for relevance classification of tweets given
a topic. Extracts classifier-related features (margin between 2 most certain
predicted classes, classifier certainty for most certain predicted class)
to be used for building the
dataset that's used for predicting label agreement.

Randomly samples 1000 tweets from each relevance class (low, medium, high) to
create 3 equisized datasets.
"""
import os
import unicodecsv as csv
import json
import random
import codecs

from emoji import UNICODE_EMOJI
import pandas as pd

from weka_functions import read_weka_predictions


# Otherwise only the first 50 chars of each column are displayed in pandas
# for debugging
pd.options.display.max_colwidth = 500


# Feature names (= column names in the data frame)
PREPROCESSED = "preprocessed"
PREPROCESSED_NO_STOPWORDS = "preprocessed_no_stopwords"
PREPROCESSED_QUERY_NO_STOPWORDS = "preprocessed_query"
RATIO_STOPWORDS = "ratio_non_stopwords"
TEXT = "text"
VOTES = "votes"
COSTS = "labeling_cost"
SENTIMENT = "watson_sentiment"
NER_WATSON = "watson_ners"
NER_ROSETTE = "rosette_ners"
KEYWORDS = "keywords"
LENGTH = "length"
QUERY = "query"
COSINE = "cos_sim"
RA_PROP = "ra_prop"
POS_TAG = "pos_tags"
HASH = "has_hash"
URL = "has_url"
MENTION = "mentions"
NER_NUMBER = "#ners"
EXPANDED_QUERY = "expanded_query"
EXPANDED_PREPROCESSED_NO_STOPWORDS = "expanded_preprocessed_no_stopwords"
DIVERSITY = "diversity"
EXTRA_TERMS = "extra_terms"
OVERLAPPING_TERMS = "overlapping_terms"
MISSING_TERMS = "missing_terms"
WEIRD = "weird"
LCS = "lcs"
DOT = "dot"
JARO_WINKLER = "jaro_winkler"

# Label name for regression
LABEL_REG = "agree_score"
# Label name for classification
LABEL_CLA = "agreement"


# Discretize agreement score for classification
# According to /media/data/Workspaces/PythonWorkspace/phd/
# Analyze-Labeled-Dataset/www2018_results/figures/
# dataset_min_annos_3_agreement_score_distribution_cleaned.pdf
BINS = [0, 0.45, 0.85, 1.0]
# Corresponding labels for the intervals
AGREEMENT_LEVELS = ["low", "medium", "high"]

# Seed for randomization
SEED = 13

# Prefix added in front of feature names to indicate that it's part of BoW
# representation
BOW_PREFIX = "bow_"
# Prefix added in front of feature names to indicate that it's part of BoW
# representation build from keywords
KEY_PREFIX = "key_"
# Prefix used for features of a glove vector
GLOVE_PREFIX = "glove_"

# BoW vectorizer (used to convert text into matrix representation) is
# trained on training set and then applied to test set
BOW_VECTORIZER = None
# KEY_VECOTRIZER = None
# Stores the TF-IDF weight for a word of the dataset
# {word1: tfidf1, word2: tfidf2,...}
TFIDF_WEIGHTS = None

# List of features that are used for training a classifier - all features
# ending with _PREFIX serve as placeholders and are replaced in the code
FEATURES = [SENTIMENT, RATIO_STOPWORDS, KEY_PREFIX,
            GLOVE_PREFIX, COSTS, LENGTH, COSINE, RA_PROP, HASH,
            MENTION, URL, NER_NUMBER, DIVERSITY, EXTRA_TERMS, MISSING_TERMS,
            OVERLAPPING_TERMS, WEIRD, LCS, DOT, JARO_WINKLER]

# Global classifier used for optimizing classifiers with skopt (it's the only
# to pass arguments into the function at the moment...
# See here for progress:
# https://github.com/scikit-optimize/scikit-optimize/pull/500
reg = None
# Same as <reg>, but the instances
X = None
# Labels of instances
y = None

# Relevance of our Trump tweets with respect to our "query" (=some keywords)
CUSTOM_QUERY = "donald trump hillary clinton political election discussion " \
               "campaign"

random.seed(SEED)

# List of emojis that need to be replaced by HTML entities
# HTML entities from here:
# http://www.fileformat.info/info/unicode/char/1f605/index.htm
# Created manually in that HTML expressions were added manually
EMOJI_LIST = {
                 u'\U0001f480': "&#128128;",
                 u'\U0001f602': "&#128514;",
                 u'\U0001f605': "&#128517;",
                 u'\U0001f606': "&#128518;",
                 u'\U0001f389': "&#127881;",
                 u'\U0001f636': "&#128566;",
                 u'\U0001f30d': "&#127757;",
                 u'\U0001f443': "&#128067;",
                 u'\U0001f30e': "&#127758;",
                 u'\U0001f611': "&#128529;",
                 u'\U0001f610': "&#128528;",
                 u'\U0001f612': "&#128530;",
                 u'\u2615': "&#9749;",
                 u'\U0001f914': "&#129300;",
                 u'\U0001f496': "&#128150;",
                 u'\U0001f918': "&#129304;",
                 u'\xae': "&#230;",
                 u'\xa9': "&#169;",
                 u'\U0001f621': "&#128545;",
                 u'\U0001f622': "&#128546;",
                 u'\U0001f3a5': "&#127909;",
                 u'\U0001f624': "&#128548;",
                 u'\U0001f4a9': "&#128169;",
                 u'\U0001f4a8': "&#128168;",
                 u'\U0001f52b': "&#128299;",
                 u'\U0001f4aa': "&#128170;",
                 u'\U0001f62d': "&#128557;",
                 u'\U0001f62c': "&#128556;",
                 u'\U0001f4af': "&#128175;",
                 u'\U0001f62e': "&#128558;",
                 u'\U0001f631': "&#128561;",
                 u'\U0001f649': "&#128585;",
                 u'\U0001f633': "&#128563;",
                 u'\U0001f40d': "&#128013;",
                 u'\U0001f637': "&#128567;",
                 u'\U0001f648': "&#128584;",
                 u'\U0001f438': "&#128056;",
                 u'\U0001f6bd': "&#128701;",
                 u'\U0001f440': "&#128064;",
                 u'\U0001f643': "&#128579;",
                 u'\U0001f642': "&#128578;",
                 u'\U0001f645': "&#128581;",
                 u'\U0001f644': "&#128580;",
                 u'\U0001f4c9': "&#128201;",
                 u'\U0001f64d': "&#128589;",
                 u'\U0001f64a': "&#128586;",
                 u'\U0001f44d': "&#128077;",
                 u'\U0001f64c': "&#128588;",
                 u'\U0001f44f': "&#128079;",
                 u'\U0001f64f': "&#128591;",
                 u'\U0001f525': "&#128293;",
                 u'\U0001f3fb': "&#127995;",
                 u'\U0001f3a4': "&#127908;",
                 u'\U0001f62a': "&#128554;",
                 u'\u2764': "&#10084;",
                 u'\U0001f469': "&#128105;",
                 u'\U0001f62f': "&#128559;",
                 u'\U0001f471': "&#128113;",
                 u'\U0001f629': "&#128553;",
                 u'\U0001f377': "&#127863;",
                 u'\u27a1': "&#10145;",
                 u'\U0001f37b': "&#127867;",
                 u'\U0001f3fd': "&#127997;",
                 u'\U0001f3fc': "&#127996;",
                 u'\U0001f3fe': "&#127998;"
}


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


def create_csv(tweets, dst, size):
    """
    Creates a csv file containing <size> tweets.

    Parameters
    ----------
    tweets: dict -
    {
        tid:
            {
                "text": "...",
                "query": "..."
            }
    }
    dst: str - path where csv file will be stored.
    size: int - number of tweets per crowdsourcing dataset.

    """
    tids = random.sample(tweets.keys(), size)
    with open(dst, "wb") as f:
        writer = csv.writer(f, dialect='excel', encoding='utf-8', )
        # Write header
        f.write("tweetID,query,tweet\n")
        for tid in tids:
            query = tweets[tid]["query"]
            text = tweets[tid]["text"]
            writer.writerow([tid, query, text])


def create_datasets(tweets, preds, dst_dir, size):
    """
    Creates 3 crowdsourcing datasets in .csv format. Each one contains <size>
    tweets which are selected randomly. One dataset contains only tweets
    with predicted low agreement, the next one only tweets with predicted
    medium agreement, and the last one only tweets with predicted high
    agreement.

    Parameters
    ----------
    tweets: dict - {
                        tid:
                        {
                            "text": "...",
                            "query": "...",
                            ...
                        }
                    }
    preds: dict - {
                        tid:
                        {
                            "label": "...",
                            ...
                        }
    dst_dir: str - directory in which the crowdsourcing datasets should be
    stored.
    size: int - number of tweets per crowdsourcing dataset.

    """
    # {
    #     tid:
    #         {
    #             "text": "...",
    #             "query": "..."
    #         }
    # }
    low = {}
    medium = {}
    high = {}
    # Group predicted labels according to agreement level
    for tid in preds:
        label = preds[tid]["label"]
        if label == "low":
            low[tid] = {
                "text": tweets[tid]["text"],
                "query": CUSTOM_QUERY
            }

        if label == "medium":
            medium[tid] = {
                "text": tweets[tid]["text"],
                "query": CUSTOM_QUERY
            }
        if label == "high":
            high[tid] = {
                "text": tweets[tid]["text"],
                "query": CUSTOM_QUERY
            }
    print "#low", len(low)
    print "#medium", len(medium)
    print "#high", len(high)
    fname = "low.csv"
    dst = os.path.join(dst_dir, fname)
    create_csv(low, dst, size)

    fname = "medium.csv"
    dst = os.path.join(dst_dir, fname)
    create_csv(medium, dst, size)

    fname = "high.csv"
    dst = os.path.join(dst_dir, fname)
    create_csv(high, dst, size)


def replace_emoji_by_html(dst_dir):
    """
    Replaces UTF-8 emoticons by HTML code and stores results in new csv files.

    Parameters
    ----------
    dst_dir: str - path to csv files.


    """
    # For each csv file
    for fn in os.listdir(dst_dir):
        texts = []
        tids = []
        queries = []
        # Ignore folders
        src = os.path.join(dst_dir, fn)
        if os.path.isfile(src):
            # Read file
            with open(src, "rb") as f:
                reader = csv.reader(f)
                # For each line, but skip header
                for idx, row in enumerate(reader):
                    # Skip header
                    if idx > 0:
                        # For each emoji
                        for emo in UNICODE_EMOJI:
                            if emo in row[2]:
                                if emo not in EMOJI_LIST:
                                    EMOJI_LIST[emo] = None
                        # Store data for potentially replacing it
                        tids.append(row[0])
                        queries.append(row[1])
                        texts.append(row[2])

            # Replace emojis and store results in new file
            fnew = fn.split(".")[0] + "_cleaned.csv"
            dst = os.path.join(dst_dir, fnew)
            print "store under", dst
            with open(dst, "wb") as f:
                writer = csv.writer(f, dialect='excel', encoding='utf-8', )
                # Write header
                f.write("tweetID,query,tweet\n")
                for idx, tid in enumerate(tids):
                    query = queries[idx]
                    text = texts[idx]
                    print "text", text
                    for emo in EMOJI_LIST:
                        pass
                        if emo in text:
                            print "text replaced"
                            text = text.replace(emo, EMOJI_LIST[emo])
                            print text
                    writer.writerow([tid, query, text])

    print "emoticons"
    # DON'T DELETE - this allows building EMOJI_LIST
    print EMOJI_LIST.keys()
    print "#unique emojis", len(EMOJI_LIST)


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Directory in which the full Twitter dataset is stored
    FULL_DIR = os.path.join(base_dir, "results", "dataset_twitter_full")
    # Directory in which the predictions are stored
    WEKA_DIR = os.path.join(base_dir, "results", "weka_predictions",
                            "agreement_predictions_twitter")
    # Directory in which dataset will be stored
    DS_DIR = os.path.join(base_dir, "results", "dataset_twitter")
    # Directory in which the resulting datasets will be stored
    DST_DIR = os.path.join(base_dir, "results", "crowdsourcing_datasets")

    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)

    # Number of randomly selected tweets per dataset
    DATASET_SIZE = 1000

    fname = "agreement_predictions_test_set.csv"
    pred_src = os.path.join(WEKA_DIR, fname)

    src = os.path.join(FULL_DIR, "dataset_twitter_full_watson_rosette_"
                                 "cleaned.json")

    # Read dataset to get texts and queries
    tweets = read_gizem_json(src)

    # Read predictions to create clusters of data
    preds = read_weka_predictions(pred_src)

    # create_datasets(tweets, preds, DST_DIR, DATASET_SIZE)

    # PROBLEM: AMT doesn't accept UTF-8 bytes > 3, i.e. no emoticons...
    # So Gizem deleted them on her end
    # replace_emoji_by_html(DST_DIR + "deb/")

    # with open("/media/data/Workspaces/PythonWorkspace/phd/Analyze-Labeled-Dataset/www2018_results/crowdsourcing_datasets/low.csv", "rb") as f:
    #     reader = csv.reader(f)
    #     for row in reader:
    #         print len(row)
