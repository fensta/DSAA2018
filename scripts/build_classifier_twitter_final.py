"""
Builds a dataset from the Twitter dataset suitable for
predicting tweet agreement for tweets by adding 2 classifier-related features
from Weka. Builds a training set and a test set (full Twitter).
Adds to both datasets the classifier-related features which were obtained
by using 10-fold CV on the training set and the full Twitter set as a test set
for the small Twitter training set.
"""
import os
import re
import codecs
import ntpath
import json
import math
import copy
from collections import Counter

import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import metrics
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import gensim
from skopt import gp_minimize
from sklearn import preprocessing
from pyjarowinkler import distance


# Otherwise only the first 50 chars of each column are displayed in pandas
# for debugging
pd.options.display.max_colwidth = 500


# Feature names (= column names in the data frame)
PREPROCESSED = "preprocessed"
PREPROCESSED_NO_STOPWORDS = "preprocessed_no_stopwords"
PREPROCESSED_QUERY_NO_STOPWORDS = "preprocessed_query"
RATIO_STOPWORDS = "ratio_non_stopwords"
TEXT = "text"
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
REL_CERTAIN = "clf_certainty_rel"
REL_MARGIN = "clf_margin_rel"
SEN_CERTAIN = "clf_certainty_sen"
SEN_MARGIN = "clf_margin_sen"
MARG = "margin"
TOP_PROBAS = "top_k_probas"

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
# See <BOW_PREFIX>, but now for TF-IDF
# TFIDF_PREFIX = "tfidf_"
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

# TF-IDF vectorizer (used to convert text into matrix representation) is
# trained on training set and then applied to test set
# TFIDF_VECTORIZER = None


# Names of columns in the data frame that don't represent training features
# List of all features in dataset
# ALL_COLS = [PREPROCESSED, PREPROCESSED_NO_STOPWORDS,
#                  SENTIMENT, RATIO_STOPWORDS, NER_WATSON,
#                  NER_ROSETTE, KEYWORDS, PREPROCESSED,
#                  PREPROCESSED_NO_STOPWORDS, TEXT, VOTES, LABEL]

# List of features that are used for training a classifier - all features
# ending with _PREFIX serve as placeholders and are replaced in the code
###########################################################################
# IMPORTANT: DON'T ADD <BOW_PREFIX> because then the computer will run out
# of RAM (>20GB) and BOW isn't used anyway
########################################################
# List of features that are used for training a classifier - all features
# ending with _PREFIX serve as placeholders and are replaced in the code
FEATURES = [SENTIMENT, RATIO_STOPWORDS, KEY_PREFIX,
            GLOVE_PREFIX,
            # COSTS,  # only available in training set
            LENGTH, COSINE, RA_PROP,
            # HASH, # every tweet has hashtag
            # BOW_PREFIX,
            # MENTION,  # almost no tweet has one
            # URL, # no tweet contains URLs that's why they were chosen
            NER_NUMBER, DIVERSITY, EXTRA_TERMS, MISSING_TERMS,
            OVERLAPPING_TERMS, WEIRD, LCS, DOT, JARO_WINKLER
            # SEN_MARGIN, SEN_CERTAIN, REL_CERTAIN, REL_MARGIN  # don't use as
            # there's no time to make predictions for test set
            ]

# Names of Gizem's additionally extracted features
# Some Features that have all the same value and are thus ignored
GIZEMS_NAMES = ['tweet_AvgPol_Twitter', 'tweet_AvgPol_TwitterRatio',
                 # 'tweet_minPolToken_Twitter',
                 'tweet_maxPolToken_Twitter',
                 'tweetFirst_AvgPol_Twitter', 'tweetFirst_AvgPol_TwitterRatio',
                 # 'tweetFirst_minPolToken_Twitter',
                 'tweetFirst_maxPolToken_Twitter', 'tweetSecond_AvgPol_Twitter',
                 'tweetSecond_AvgPol_TwitterRatio',
                 # 'tweetSecond_minPolToken_Twitter',
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
                 'tweet_PosTermRatio', 'tweet_NegTermRatio',
                 # 'tweetFirstLength',
                 # 'tweetSecondLength',
                 'tweetFirst_PosWords',
                 'tweetFirst_PosTermRatio', 'tweetFirst_NegWords',
                 'tweetFirst_NegTermRatio', 'tweetSecond_PosWords',
                 'tweetSecond_PosTermRatio', 'tweetSecond_NegWords',
                 'tweetSecond_NegTermRatio', 'tweet_SumFreq', 'tweet_MeanFreq',
                 'tweet_MinFreq', 'tweet_MaxFreq', 'tweet_VarianceFreq',
                 'tweet_NounsNum', 'tweet_AdjsNum', 'tweet_AdvsNum',
                 'tweet_VerbsNum', 'tweet_NounPercentage',
                 'tweet_AdjPercentage', 'tweet_AdvPercentage',
                 'tweet_VerbPercentage',
                 # 'numQueryOccurrenceInTweet',
                 'tweet_numExcMarks', 'tweet_numQuestMarks',
                 'tweet_numSuspensionPoints', 'tweet_numQuotationMarks',
                 'tweet_numKeywordWould', 'tweet_numKeywordLike',
                 # 'tweet_numKeywordSudden',
                 'tweet_numKeywordYet',
                 'tweet_numTwitterLingos',
                 # 'tweet_numPosEmots',
                 # 'tweet_numNegEmots',
                 'tweet_numExcMarksRatio',
                 'tweet_numQuestMarksRatio',
                 # 'tweet_numPosEmotsRatio',
                 # 'tweet_numNegEmotsRatio',
                 'tweet_numAllUppercaseTokens',
                 'tweet_numAllUppercaseTokensRatio',
                 'tweet_numRepeatingCharactersTokens',
                 'tweet_numRepeatingCharactersTokensRatio',
                 'levenshteinDistance', 'jaccardSimilarityShingle1',
                 'jaccardSimilarityShingle2', 'tweet_probabilityLDATopic1',
                 'tweet_probabilityLDATopic2', 'tweet_probabilityLDATopic3',
                 'tweet_probabilityLDATopic4', 'tweet_probabilityLDATopic5',
                 'tweet_probabilityLDATopic6', 'tweet_probabilityLDATopic7',
                 'tweet_probabilityLDATopic8', 'tweet_probabilityLDATopic9',
                 'tweet_probabilityLDATopic10'
                ]
FEATURES.extend(GIZEMS_NAMES)

# Relevance of our Trump tweets with respect to our "query" (=some keywords)
CUSTOM_QUERY = "donald trump hillary clinton political election discussion " \
               "campaign"

# Global classifier used for optimizing classifiers with skopt (it's the only
# to pass arguments into the function at the moment...
# See here for progress:
# https://github.com/scikit-optimize/scikit-optimize/pull/500
reg = None
# Same as <reg>, but the instances
X = None
# Labels of instances
y = None


######################
# Feature extraction #
######################
def add_features(df, is_training_set=True, glove_src="", use_tfidf=False):
    """
    Adds columns to the data frame that contains multiple features.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.
    is_training_set: bool - True if it's the training set that should be
    preprocessed. Otherwise the test set will be preprocessed which uses
    existing statistics from the training set.
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's left empty, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly. Only considered if
    <glove_src> isn't empty.

    Returns
    -------
    pandas.DataFrame.
    Dataset with additional columns for the extracted features.

    """
    global TFIDF_WEIGHTS

    df = add_fraction_stopwords(df)

    # Use tweet length
    df = add_tweet_length(df)
    # Add labels for training set in classification task
    if is_training_set:
        df[LABEL_CLA] = pd.cut(df[LABEL_REG], BINS, labels=AGREEMENT_LEVELS)
    else:
        # Test set labels must be predicted
        df[LABEL_CLA] = "?"
    # TODO: BOW is disabled to save RAM
    if is_training_set:
        # df = add_bow_for_training_set(df)
        # Always compute TF-IDF weights because they might be used by other
        # features for weighting, such as cosine similarity
        TFIDF_WEIGHTS = get_tfidf(df)
        # Export TF-IDF for Gizem
        # import json
        #
        # with codecs.open("result.json", "w") as fp:
        #     data = json.dumps(TFIDF_WEIGHTS, encoding="utf-8",
        #                       ensure_ascii=False)
        #     fp.writelines(unicode(data))
    else:
        # Apply existing vectorizer to transform test set according to training
        # set
        # df = add_bow_for_test_set(df)
        pass

    # Use Glove vectors
    if len(glove_src) > 0:
        vecs = read_glove_vectors(glove_src)
        # Weigh glove vectors according to TF-IDF weights
        if use_tfidf:
            df = add_glove_representation(df, vecs, TFIDF_WEIGHTS)
        # Weigh glove vectors uniformly
        else:
            df = add_glove_representation(df, vecs)

    # Add features related to the paper of Tao et al.
    df = add_tao(df)

    # IMPORTANT: compute similarities between query and text AFTER add_tao()
    # because this potentially expands the query and tweet text terms
    # (commented out atm though), so the comment is currently irrelevant

    # Add cosine similarity between query and tweet
    df = add_cosine_similarity(df)

    # Add similarity between query and tweet according to a paper
    df = add_ra_prop(df)

    # Add features from Dong et al.
    df = add_dong(df)

    # Add longest common subsequence
    df = add_lcs(df)

    # Add Jaro-Winkler
    df = add_jaro_winkler(df)

    # Add classifier-related features
    df = add_clf_features(df)

    return df


def add_clf_features(df):
    """
    Adds classifier-related features, namely certainty of classifier for
    most likely predicted label and margin between this certainty and
    certainty for 2nd most likely class label.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Dataset with additional columns for the features.
    Adds the columns <MARGIN> and <CERTAIN>.

    """
    df = df.apply(clf_features, axis=1)
    return df


def clf_features(row):
    """
    Adds classifier-related features, namely certainty of classifier for
    most likely predicted label and margin between this certainty and
    certainty for 2nd most likely class label.

    Parameters
    ----------
    row: pandas.Series - row in the dataset representing a tweet.

    Returns
    -------
    pandas.DataFrame.
    Columns with the respective features.

    """
    # 1. Tokenize text by splitting on whitespaces
    query_words = row[PREPROCESSED_QUERY_NO_STOPWORDS].split()
    text_words = row[PREPROCESSED_NO_STOPWORDS].split()
    comparisons = 0
    jaro_winkler_sim = 0.0

    for qw in query_words:
        for tw in text_words:
            jaro_winkler_sim += \
                distance.get_jaro_distance(tw, qw, winkler=True, scaling=0.1)
            comparisons += 1

    # Average similarity per word-to-word comparison
    row[JARO_WINKLER] = 1.0*jaro_winkler_sim / comparisons
    return row


def add_jaro_winkler(df):
    """
    Computes Jaro-Winkler similarity between query and tweet text, i.e.
    the Jaro measure is the weighted sum of percentage of matched characters
    from each file and transposed characters and Winkler increased this measure
    for matching initial characters.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Dataset with additional columns for the features.
    Adds the column <JARO_WINKLER>.

    """
    df = df.apply(jaro_winkler, axis=1)
    return df


def jaro_winkler(row):
    """
    Adds Jaro-Winkler similarity.

    Parameters
    ----------
    row: pandas.Series - row in the dataset representing a tweet.

    Returns
    -------
    pandas.DataFrame.
    Columns with the respective features.

    """
    # 1. Tokenize text by splitting on whitespaces
    query_words = row[PREPROCESSED_QUERY_NO_STOPWORDS].split()
    text_words = row[PREPROCESSED_NO_STOPWORDS].split()
    comparisons = 0
    jaro_winkler_sim = 0.0

    for qw in query_words:
        for tw in text_words:
            jaro_winkler_sim += \
                distance.get_jaro_distance(tw, qw, winkler=True, scaling=0.1)
            comparisons += 1

    # Average similarity per word-to-word comparison
    row[JARO_WINKLER] = 1.0*jaro_winkler_sim / comparisons
    return row


def add_lcs(df):
    """
    Computes longest common subsequence between query and tweet text, i.e.
    maximum number of words that are shared in both texts in the same order.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Dataset with additional columns for the features.
    Adds the column <LCS>.

    """
    df = df.apply(longest_common_sentence, axis=1)
    return df


def lcs(s1, s2):
    """https://stackoverflow.com/questions/24547641/python-length-of-longest-common-subsequence-of-lists"""
    table = [[0] * (len(s2) + 1) for _ in xrange(len(s1) + 1)]
    for i, ca in enumerate(s1, 1):
        for j, cb in enumerate(s2, 1):
            table[i][j] = (
                table[i - 1][j - 1] + 1 if ca == cb else
                max(table[i][j - 1], table[i - 1][j]))
    return table[-1][-1]


def longest_common_sentence(row):
    """
    Computes the longest common subsequence (= words that occur in both
    sentences in the same order, but not necessarily in contiguous order)
    between two sentences s1 and s2.
    IMPORTANT: returns normalized number of matches, but it can be at most
    as long as the shorter sentence (here: query), thus, divide by number of
    words in shorter sentence, i.e., distance is within (and including) [0,1].

    Parameters
    ----------
    row: pandas.Series - row in the dataset representing a tweet.

    Returns
    -------
    Normalized LCS distance. Normalization is implemented by dividing
    LCS / min(len(s1, s2), i.e. distance is within (and including) [0,1].

    """
    query = row[PREPROCESSED_QUERY_NO_STOPWORDS]
    text = row[PREPROCESSED_NO_STOPWORDS]
    words = lcs(query, text)
    row[LCS] = 1.0 * words / min(len(query), len(text))
    return row


def add_dong(df):
    """
    Adds some of the features proposed in "Time is of the Essence: Improving
    Recency Ranking Using Twitter Data" by Dong et al.
    We focus on #overlapping terms, #missing terms, #extra terms.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Dataset with additional columns for the features.
    Adds the columns <OVERLAPPING_TERMS>, <MISSING_TERMS>, <EXTRA_TERMS>.

    """
    df = df.apply(dong, axis=1)
    return df


def dong(row):
    """
    Adds a couple of features described in the paper.

    Parameters
    ----------
    row: pandas.Series - row in the dataset representing a tweet.

    Returns
    -------
    pandas.DataFrame.
    Columns with the respective features.

    """
    # 1. Tokenize text by splitting on whitespaces
    query_words = row[PREPROCESSED_QUERY_NO_STOPWORDS].split()
    text_words = row[PREPROCESSED_NO_STOPWORDS].split()
    q = set(query_words)
    t = set(text_words)
    overlapping = len(q & t)
    extra = len(t) - overlapping
    missing = len(q) - overlapping
    row[MISSING_TERMS] = missing
    row[OVERLAPPING_TERMS] = overlapping
    row[EXTRA_TERMS] = extra
    # This equation doesn't exist in the paper, but let's try it out
    row[WEIRD] = 1.0/len(q) * (extra**0.5 + missing**0.65 + overlapping)*len(t)
    return row


def add_tao(df):
    """
    Adds some of the features proposed in "What makes a tweet relevant for a
    topic?" by Tao et al.
    We focus on hasURL, #entities, diversity, hasHashtag and also add #mentions.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Dataset with additional columns for the features.
    Adds the columns <HASH>, <URL>, <MENTION>, <NER_NUMBER>, <DIVERSITY>,
    <EXPANDED_QUERY>, <EXPANDED_TEXT_NO_STOPWORDS>.

    """
    df = df.apply(tao, axis=1)
    return df


def tao(row):
    """
    Adds a couple of features described in the paper.

    Parameters
    ----------
    row: pandas.Series - row in the dataset representing a tweet.

    Returns
    -------
    pandas.DataFrame.
    Columns with the respective features.

    """
    has_url = False
    has_mention = False
    has_hashtag = False

    # 1. Tokenize text by splitting on whitespaces
    query_words = row[PREPROCESSED_QUERY_NO_STOPWORDS].split()
    # Use original text to find @ and # and so on
    text_words = row[TEXT].split()

    # 2. Extract features related to text
    for word in text_words:
        if word.startswith("#"):
            has_hashtag = True
        if word.startswith("http"):
            has_url = True
        if word.startswith("@"):
            has_mention = True

    # 3. Merge NERs identified by Rosette and Watson
    # {NER1: ([more abstract terms] OR None, [list of NER types])}
    ners = {}
    # {NER1: (None, NER type)}
    ros_ners = {ner_n: (None, ty) for (ner_t, ner_n, ty) in row[NER_ROSETTE]}
    # {NER1: ([more abstract terms] OR None, NER type)}
    wat_ners = {ner_n: (dis, ty) for (ner_n, dis, ty) in row[NER_WATSON]}
    # Get a list of all unique NERs
    ner_words = set(ros_ners.keys() + wat_ners.keys())
    # print "ner words"
    # print ner_words
    for ner in ner_words:
        # print "ner:", ner
        # Merge both entries
        if ner in ros_ners and ner in wat_ners:
            _, ros_type = ros_ners[ner]
            disamb, wat_type = wat_ners[ner]
            # Disambiguation only available in Watson, but not Rosette
            # Merge NER types (each API assigns a single one only)
            ners[ner] = (disamb, list({ros_type} | {wat_type}))
            # Expand text with abstract terms
            # if disamb is not None:
            #     text_words.extend(disamb)
            #     # Expand query with abstract terms
            #     for w in query_words:
            #         if w in disamb:
            #             print "query before", text_words
            #             query_words.extend(disamb)
            #             print "query after", text_words

        # Use the entry from Rosette
        if ner in ros_ners and ner not in wat_ners:
            ners[ner] = (ros_ners[ner][0], [ros_ners[ner][1]])
            # print "ROSETTE", ros_ners[ner]
        # Use the entry from Watson
        if ner in wat_ners and ner not in ros_ners:
            disamb = wat_ners[ner][0]
            ners[ner] = (disamb, [wat_ners[ner][1]])
            # print "WATSON", wat_ners[ner]
            # Expand text with abstract terms
            # if disamb is not None:
            #     text_words.extend(disamb)
            #     # Expand query with abstract terms
            #     for w in query_words:
            #         if w in disamb:
            #             print "query before", text_words
            #             query_words.extend(disamb)
            #             print "query after", text_words
        # print "MERGED", ners[ner]

    # Count unique NER types for text
    unique_ners = set([ty for ner in ners for ty in ners[ner][1]])

    # 4. Add features to tweet
    if has_url:
        row[URL] = 1
    else:
        row[URL] = 0
    if has_mention:
        row[MENTION] = 1
    else:
        row[MENTION] = 0
    if has_hashtag:
        row[HASH] = 1
    else:
        row[HASH] = 0
    row[DIVERSITY] = len(unique_ners)
    row[NER_NUMBER] = len(ners)

    return row


def add_ra_prop(df):
    """
    Computes the similarity between query and tweet text according to
    "RAProp: Ranking Tweets by Exploiting the Tweet/User/Web Ecosystem and
    Inter-Tweet Agreement" by S. Ravikumar et al., but only uses TF-IDF part

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Dataset with additional column for the cosine similarity.

    """
    df = df.apply(ra_prop, axis=1)
    return df


def ra_prop(row):
    """
    Adds similarity between query and tweet similar to RaProp.

    Parameters
    ----------
    row: pandas.Series - row in the dataset representing a tweet.

    Returns
    -------
    pandas.Series.
    Updated row with cosine similarity between query and tweet text in a column
    called <RA_PROP>.

    """
    # ADJ: Adjective
    # ADP: Adposition
    # ADV: Adverb
    # AUX: Auxiliary verb
    # CONJ: Coordinating conjunction
    # DET: Determiner
    # INTJ: Interjection
    # NOUN: Noun
    # NUM: Numeral
    # PART: Particle
    # PRON: Pronoun
    # PROPN: Proper noun
    # PUNCT: Punctuation
    # SCONJ: Subordinating conjunction
    # SYM: Symbol
    # VERB: Verb
    # X: Other
    weights = {
        "ADJ": 3.0,
        "ADP": 0.1,
        "ADV": 3.0,
        "AUX": 0.1,
        "CONJ": 0.1,
        "DET": 0.1,
        "INTJ": 0.5,
        "NOUN": 3.0,
        "NUM": 2.0,
        "PART": 0.1,
        "PRON": 1.0,
        "PROPN": 4.0,
        "PUNCT": 0.1,
        "SCONJ": 0.1,
        "SYM": 0.1,
        "VERB": 1.0,
        "X": 0.1,
    }
    # 1. Create vector representations for query and tweet text
    # Tokenize texts by splitting on whitespaces
    query_words_ = row[PREPROCESSED_QUERY_NO_STOPWORDS].split()
    # Use full document because Rosette POS tagger was applied to original text
    text_words_ = row[TEXT].split()

    # 2. Convert to lowercase and add POS tags
    query_words = []
    text_words = []
    for word in query_words_:
        query_words.append(word.lower())
    for word, pos in zip(text_words_, row[POS_TAG]):
        text_words.append((word.lower(), pos))

    # 3. Find words that exist in query and tweet
    intersection = []
    for word, pos in text_words:
        if word in query_words:
            intersection.append((word, pos))

    # 4. Remove stop words
    intersection = [(word, pos) for word, pos in intersection if word not in
                    stopwords.words('english')]

    score = 0.0
    # It might happen that there's no overlap at all
    if len(intersection) > 0:
        # 5. Find TF-IDF weights of each word
        tfidf = {}
        for w, _ in intersection:
            if w in TFIDF_WEIGHTS:
                tfidf[w] = TFIDF_WEIGHTS[w]
            else:
                tfidf[w] = 1.0

        # In the paper only IDF was squared and it was divided by largest TF,
        # but since both values are stored together, we use tf-idf in both cases
        score = 0.0
        for word, pos in intersection:
            # There are some tweets for which no POS tags were extracted, so
            # "None" was extracted and "None" doesn't exist in <weights>
            if pos in weights and word in tfidf:
                score += tfidf[word]**2 * weights[pos]
        # Normalize by highest tf-idf score only
        if len(tfidf) > 0:
            score /= max(tfidf.values())

    row[RA_PROP] = score
    return row


def add_cosine_similarity(df):
    """
    Computes the cosine similarity between the query and the actual tweet text
    based on which relevance for a tweet was decided.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Dataset with additional column for the cosine similarity called <COSINE>,
    and <DOT> for dot product which is unnormalized cosine similarity. The
    latter might be more suitable for Twitter:
    Normalization with tf*idf related features: If you want to stay with TF-IDF
    or its variations, you must not use length normalization techniques. We
    found that length normalization for such short texts are not only
    unmotivated but even counterproductive in {Naveed, N., Gottron, T.,
    Kunegis, J., Che Alhadi, A.: Searching microblogs: Coping with sparsity
    and document quality. }

    """
    df = df.apply(cosine_similarity, axis=1)
    return df


def cosine_similarity(row):
    """
    Computes the cosine similarity for a row in a pandas.DataFrame.

    Parameters
    ----------
    row: pandas.Series - row in the dataset representing a tweet.

    Returns
    -------
    pandas.Series.
    Updated row with cosine similarity between query and tweet text in a column
    called <COSINE>.

    """
    # 1. Create vector representations for query and tweet text
    # Tokenize texts by splitting on whitespaces
    query_words_ = row[PREPROCESSED_QUERY_NO_STOPWORDS].split()
    # These words were already lowercased
    text_words = row[PREPROCESSED_NO_STOPWORDS].split()
    # 2. Convert to lowercase
    query_words = []
    for word in query_words_:
        query_words.append(word.lower())
    vector1 = Counter(query_words)
    vector2 = Counter(text_words)

    # 3. Find TF-IDF weights of each word
    tf1 = {}
    for w in query_words:
        if w in TFIDF_WEIGHTS:
            tf1[w] = TFIDF_WEIGHTS[w]
        else:
            tf1[w] = 1.0
    tf2 = {}
    for w in text_words:
        if w in TFIDF_WEIGHTS:
            tf2[w] = TFIDF_WEIGHTS[w]
        else:
            tf2[w] = 1.0

    # Store result in data frame
    cos, dot = get_cosine(vector1, vector2, tf1, tf2)
    row[COSINE] = cos
    row[DOT] = dot
    return row


def get_cosine(vec1, vec2, tf1, tf2):
    """
    Computes cosine distance weighted by TF-IDF weights. See
    https://stackoverflow.com/questions/15173225/how-to-calculate-cosine-similarity-given-2-sentence-strings-python

    Parameters
    ----------
    vec1: Counter - contains words and their frequencies of the query.
    vec2: Counter - contains words and their frequencies of the tweet.
    tf1: Counter - contains TF-IDF weights for all words in <vec1>.
    tf2: Counter - contains TF-IDF weights for all words in <vec2>.

    Returns
    -------
    float, float.
    Cosine similarity, dot product (= unnormalized cosine similarity).

    """
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([tf1[x]*vec1[x] * tf2[x]*vec2[x] for x in intersection])

    sum1 = sum([(tf1[x]*vec1[x]) ** 2 for x in vec1])
    sum2 = sum([(tf2[x]*vec2[x]) ** 2 for x in vec2])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0, 0.0
    else:
        return float(numerator) / denominator, float(numerator)


def add_tweet_length(df):
    """
    Adds the length of a tweet (in characters) as a feature. Length is
    represented as a fraction of 140 characters. It's calculated over the
    original tweet.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Dataset with additional column for the tweet length.

    """
    df[LENGTH] = df[TEXT].apply(count_chars)
    return df


def add_glove_representation(df, vectors, tfidf_weights=None):
    """
    Computes the average glove vector of each tweet and adds it as a feature.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.
    vectors: dict - each word (=k) stores its corresponding glove vector.
    dataset.
    tfidf_weights: dict: Contains for each word its TF-IDF weight. If None,
    uniform weights are used for glove vectors.

    Returns
    -------
    pandas.DataFrame.
    Dataset with additional columns for the extracted features. Feature names
    will be <GLOVE_PREFIX> + "1/2/3/..n" where n is the dimensionality of the
    vectors in <glove_src>. Thus, we add n new columns.

    """
    results = []
    for text in df[PREPROCESSED_NO_STOPWORDS].tolist():
        # print "text", text
        res = compute_glove_vector_for_tweet(text, vectors, tfidf=tfidf_weights)
        # print "res dim", res.shape
        results.append(res)
    # Convert to matrix, s.t. there are len(results) x res.shape entries,
    # i.e. each res is on a separate line
    results = np.array(results)
    # Add each dimension of the glove vector as a separate column
    for col_idx in xrange(results.shape[1]):
        df[GLOVE_PREFIX + str(col_idx)] = results[:, col_idx]
    return df


def add_fraction_stopwords(df):
    """
    Adds the fraction of stopwords for each tweet as a feature accessible via
    <RATIO_STOPWORDS> in <df>.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Modified dataset with a new column (=feature)

    """
    pre_words = "preprocessed_words"
    pre_stop_words = "preprocessed_no_stopwords_words"
    # Temporary columns
    df[pre_words] = df[PREPROCESSED].apply(count_words)
    df[pre_stop_words] = df[PREPROCESSED_NO_STOPWORDS].apply(count_words)

    # Percentage ([0-1]) of stopwords in a tweet --> stopwords are easy
    # to understand, so a high percentage might correlate with
    # agreement/disagreement
    df[RATIO_STOPWORDS] = (df[pre_words] - df[pre_stop_words]) / df[pre_words]

    # Delete temporary columns
    del df[pre_words]
    del df[pre_stop_words]
    return df


def add_bow_for_training_set(df):
    """
    Adds bag-of-words representation as features where each column has the name
    of the corresponding word, prefixed with "bow_". Learns BoW vocabulary from
    training set.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Modified dataset with a new column (=feature)

    """
    # Tell Python we want to update the existing global variable instead of
    # creating a new local variable with the same name
    global BOW_VECTORIZER
    # Create bag-of-words (BoW) representation
    # Use only 1/o for presence of a word because this range is used for all
    # features
    vectorizer = CountVectorizer(binary=True)
    # Store vectorizer so that test set can be transformed later on
    BOW_VECTORIZER = vectorizer
    # {word1: column_idx_in_matrix1, ...}
    X = vectorizer.fit_transform(df[PREPROCESSED])
    # Sort words according to indices in matrix ascendingly
    # [(word1, column_idx_in_matrix1), ...]
    # tpls = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
    # words = [tpl[0] for tpl in tpls]
    words = vectorizer.get_feature_names()

    X = X.toarray()
    # print "BoW training set:", X.shape
    np.set_printoptions(threshold=np.nan)

    # Add BoW as features
    for col_idx, word in enumerate(words):
        df[BOW_PREFIX + word] = X[:, col_idx]
    return df


def add_bow_for_test_set(df):
    """
    Adds bag-of-words representation as features where each column has the name
    of the corresponding word, prefixed with "bow_". Applies vocabulary from
    training set.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    pandas.DataFrame.
    Modified dataset with a new column (=feature)

    """
    # Create bag-of-words (BoW) representation
    vectorizer = BOW_VECTORIZER
    # {word1: column_idx_in_matrix1, ...}
    X = vectorizer.transform(df[PREPROCESSED])
    # Sort words according to indices in matrix ascendingly
    # [(word1, column_idx_in_matrix1), ...]
    # tpls = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
    # words = [tpl[0] for tpl in tpls]
    words = vectorizer.get_feature_names()

    X = X.toarray()
    # print "BoW test set:", X.shape
    np.set_printoptions(threshold=np.nan)

    # Add BoW as features
    for col_idx, word in enumerate(words):
        df[BOW_PREFIX + word] = X[:, col_idx]
    return df


def get_tfidf(df):
    """
    Computes TF-IDF weights from a dataset.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.

    Returns
    -------
    dict.
    Words are keys, TF-IDF values are values.

    """
    vectorizer = TfidfVectorizer()
    # Learn weights
    X = vectorizer.fit_transform(df[PREPROCESSED])
    tfidf_weights = {}
    for word, idf in zip(vectorizer.get_feature_names(), vectorizer.idf_):
        tfidf_weights[word] = idf
    return tfidf_weights


def count_chars(text):
    """Counts characters in the string w.r.t. 140 character limit"""
    return len(text) / 140.0


def count_words(text):
    """Counts words in the string"""
    tokenized = text.split()
    return len(tokenized)


def remove_stopwords_and_lowercase(text):
    """Removes stopwords from the string"""
    tokenized = text.split()
    filtered_words = [word.lower() for word in tokenized if
                      word not in stopwords.words('english')]
    text = " ".join(filtered_words)
    return text


def compute_glove_vector_for_tweet(text, vectors, tfidf=None):
    """
    Computes a single glove vector representing a tweet. The tweet is the
    average of the separate glove vectors in a tweet's text.
    See here why it could be (not) a good idea to average vectors:
    https://stackoverflow.com/questions/29760935/how-to-get-vector-for-a-sentence-from-the-word2vec-of-tokens-in-sentence

    Parameters
    ----------
    text: str - tweet text.
    vectors: dict - contains the glove vectors (as a value, represented by an
    numpy.array) of all words in the dataset (but for some words in <text> no
    glove vector might exist).
    tfidf: dict - words are keys and TF-IDF weights are values. If not None,
    each glove vector (if it exists) is weighted by the TF-IDF weight. Otherwise
    all vectors have uniform weights.

    Returns
    -------
    numpy.array.
    Average glove vector of a tweet.

    """
    # To avoid division by 0
    number_of_vectors = 1
    # Get dimensionality of glove vectors from first arbitrary entry
    v = vectors.itervalues().next()
    result = np.zeros(shape=v.shape)

    # Tokenize text
    for word in text.split():
        weight = 1.0
        # Add a word's glove vector to the overall tweet vector
        # (Optional): weigh it by the TF-IDF weight of the word
        if word in vectors:
            # Number of glove vectors we found for the tweet
            number_of_vectors += 1
            # Use TF-IDF weight
            if tfidf is not None and word in tfidf:
                weight = tfidf[word]
            result += weight * vectors[word]
    avg_result = result / number_of_vectors
    return avg_result


def get_feature_names(df, features_):
    """
    Selects all features from a data frame that should be used. Useful, since
    some features aren't present yet, only their indicators that they should
    be used.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.
    features_: list of str - names of the features (=columns) in <df>.

    Returns
    -------
    List of str.
    Names of all features to be used.

    """
    # Don't make changes to <features> as it would affect the original variable
    features = copy.deepcopy(features_)

    # Add BoW, Glove, and keywords if specified
    # Use BoW features - remove the prefix from features and add actual BoW
    # words
    try:
        features.remove(BOW_PREFIX)
        print "use BoW"
        add = [c for c in df.columns if c.startswith(BOW_PREFIX)]
        print "add {} features".format(len(add))
        features.extend(add)
    except ValueError:
        print "no bow"
        pass
    # Use same logic to add/don't add keywords
    try:
        features.remove(KEY_PREFIX)
        print "use keywords"
        add = [c for c in df.columns if c.startswith(KEY_PREFIX)]
        print "add {} features".format(len(add))
        features.extend(add)
    except ValueError:
        print "no keywords"
        pass

    # Use same logic to add/don't add glove vector representation
    try:
        features.remove(GLOVE_PREFIX)
        print "use glove"
        add = [c for c in df.columns if c.startswith(GLOVE_PREFIX)]
        print "add {} features".format(len(add))
        features.extend(add)
    except ValueError:
        print "no glove"
        pass
    return features


################
# Input/output #
################
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


def read_as_df(src):
    """
    Reads in the dataset into a data frame object.

    Parameters
    ----------
    src: str - path to dataset.

    Returns
    -------
    pandas.DataFrame.
    Column names: text, agree_score, votes (= #annotators who labeled it),
    labeling_cost

    """
    tweets = read_json(src)
    tids = []
    texts = []
    sentiments = []
    watson_ners = []
    rosette_ners = []
    keywords = []
    pos_tags = []
    scores = []
    costs = []
    for tid in tweets:
        tids.append(tid)
        t = tweets[tid]
        scores.append(t[LABEL_REG])
        texts.append(t[TEXT])
        costs.append(t[COSTS])
        rosette_ners.append(t[NER_ROSETTE])
        watson_ners.append(t[NER_WATSON])
        pos_tags.append(t[POS_TAG])
        sentiments.append(t[SENTIMENT])
        keywords.append(t[KEYWORDS])
    data = {
        TEXT: texts,
        LABEL_REG: scores,
        SENTIMENT: sentiments,
        NER_WATSON: watson_ners,
        NER_ROSETTE: rosette_ners,
        KEYWORDS: keywords,
        QUERY: CUSTOM_QUERY,
        COSTS: costs,
        POS_TAG: pos_tags
    }
    df = pd.DataFrame(index=tids, data=data)
    return df


def read_as_df_gizem(src, src_pred_rel, src_pred_sen):
    """
    Reads in the dataset into a data frame object. Same as read_as_df(), but
    adds Gizem's extracted features as well

    Parameters
    ----------
    src: str - path to dataset.
    src_pred_rel: str - path to Weka's predictions for relevance.
    src_pred_rel: str - path to Weka's predictions for sentiment.

    Returns
    -------
    pandas.DataFrame.
    Column names: text, agree_score, votes (= #annotators who labeled it),
    labeling_cost

    """
    # TODO: comment in if predictions should be added
    # rel_preds = get_top_k_weka_predictions(src_pred_rel, 2, binary=True)
    # sen_preds = get_top_k_weka_predictions(src_pred_sen, 2)
    sen_preds = {}
    rel_preds = {}
    tweets = read_json(src)
    tids = []
    texts = []
    sentiments = []
    watson_ners = []
    rosette_ners = []
    keywords = []
    pos_tags = []
    rel_certains = []
    rel_margins = []
    sen_certains = []
    sen_margins = []
    # costs = []
    scores = []
    # List of lists. The i-th inner list represents her i-th feature according
    # to <GIZEMS_NAMES>.
    gizems_features = [[] for _ in xrange(len(GIZEMS_NAMES))]
    for tid in tweets:
        tids.append(tid)
        t = tweets[tid]
        # Get certainty of highest class
        if tid in rel_preds:    # Always true if rel_preds is created
            rel_most_certain = rel_preds[tid][TOP_PROBAS][0]
            rel_certains.append(rel_most_certain)
            rel_margins.append(rel_preds[tid][MARG])

        # Only tweets that were predicted as relevant, have sentiment
        # predictions
        if tid in sen_preds:
            sen_most_certain = sen_preds[tid][TOP_PROBAS][0]
            sen_certains.append(sen_most_certain)
            sen_margins.append(sen_preds[tid][MARG])
        else:
            sen_certains.append(0.0)
            sen_margins.append(0.0)
        # Only true if it's a training set
        if LABEL_REG in t:
            scores.append(t[LABEL_REG])
        else:
            # Test set
            scores.append(0)
        # scores.append(t[LABEL_REG])
        texts.append(t[TEXT])
        rosette_ners.append(t[NER_ROSETTE])
        watson_ners.append(t[NER_WATSON])
        pos_tags.append(t[POS_TAG])
        # costs.append(t[COSTS])
        sentiments.append(t[SENTIMENT])
        keywords.append(t[KEYWORDS])
        # Add Gizem's features
        for idx, name in enumerate(GIZEMS_NAMES):
            gizems_features[idx].append(t[name])
    data = {
        TEXT: texts,
        SENTIMENT: sentiments,
        NER_WATSON: watson_ners,
        NER_ROSETTE: rosette_ners,
        KEYWORDS: keywords,
        POS_TAG: pos_tags,
        # TODO: uncomment to have these available
        # REL_CERTAIN: rel_certains,
        # REL_MARGIN: rel_margins,
        # SEN_CERTAIN: sen_certains,
        # SEN_MARGIN: sen_margins,
        QUERY: CUSTOM_QUERY,
    }
    # Store labels only in training set
    if len(scores) == len(tweets):
        data[LABEL_REG] = scores
    # if len(costs) == len(tweets):
    #     data[COSTS] = costs
    #     data[VOTES] = votes
    for idx, f in enumerate(gizems_features):
        feature_name = GIZEMS_NAMES[idx]
        data[feature_name] = gizems_features[idx]
    df = pd.DataFrame(index=tids, data=data)
    return df


def read_glove_vectors(src):
    """
    Reads in the glove vectors from a json file.

    Parameters
    ----------
    src: str - path to the glove vector file.

    Returns
    -------
    dict.
    {word: np.array() representing a glove vector}.
    Glove vectors.

    """
    with codecs.open(src, "r", encoding="utf-8") as f:
        vecs = json.load(f, encoding="utf-8")
    # Convert the lists of values representing a glove vector into a numpy array
    for word in vecs:
        vecs[word] = np.array(vecs[word])
    return vecs


#################
# Preprocessing #
#################
def clean_tweets(df):
    """
    Preprocesses the tweets.

    Parameters
    ----------
    df: pandas.DataFrame - holds the data with column names "text",
    "agree_score", "votes".

    Returns
    -------
    pandas.DataFrame.
    Updated data frame with preprocessed tweet texts in a new column called
    "preprocessed".

    """
    df[PREPROCESSED] = df[TEXT].apply(preprocess)
    # Same as "preprocessed", but without stop words
    df[PREPROCESSED_NO_STOPWORDS] = df[PREPROCESSED].apply(
        remove_stopwords_and_lowercase)
    df[PREPROCESSED_QUERY_NO_STOPWORDS] = df[QUERY].apply(
        remove_stopwords_and_lowercase)
    return df


def preprocess(text):
    """
    Applies various preprocessing steps to each tweet text.

    Parameters
    ----------
    text: text of a tweet to be preprocessed.

    Returns
    -------
    str.
    Preprocessed tweet text.

    """
    # Keep only letters in a string and replace the rest with a whitespace
    text = re.sub("[^a-zA-Z]", " ", text)
    # Remove all double (or more) whitespaces
    text = " ".join(text.split())
    return text


###########################
# Training and evaluation #
###########################
def get_x_and_y(data, features_, label, normalize):
    """
    Obtains a matrix representation of the instances using the selected
    features. Also returns the corresponding labels of these instances.

    Parameters
    ----------
    data: pandas.DataFrame - instances which should be transformed into a
    matrix representation.
    features_: list of str - representing the names of the features (=columns) in
    <data> that should be used for building the classifier.
    label: str - name of the column in <data> that represents the
    the label.
    normalize: bool - True if each column should be in 0-1 range (borders
    included)

    Returns
    -------
    numpy.array, list.
    Matrix representation of data frame using only the specified features.
    Labels of the instances in a 1d array.

    """
    # We're potentially removing elements from the list, so it'll be updated
    # in the parent function as well which isn't what we want - we want it to
    # remain the same
    features = get_feature_names(data, features_)

    # Get labels, i.e. agreement scores
    y = data[label].tolist()
    # print "#features + labels training set", train.shape
    # Use specified features
    # features = [c for c in train.columns if c not in EXCLUDED_COLS]
    print "use features:", features
    if normalize:
        # Put the data into [0...1] (including borders) range
        minmax_scale = preprocessing.MinMaxScaler().fit(data[features])
        x = minmax_scale.transform(data[features])
        print x.shape
    else:
        x = data.as_matrix(columns=features)
    return x, y


def train_and_evaluate(train, test, clf, features, label, normalize):
    """
    Builds a classifier and evaluates it.

    Parameters
    ----------
    train: pandas.DataFrame - training set.
    test: pandas.DataFrame - test set.
    clf: sklearn classifier - classifier to be trained.
    features: list of str - representing the names of the features (=columns) in
    <train> and <test> that should be used for building the classifier.
    label: str - name of the column in <train> and <test> that represents the
    the label.
    normalize: bool - True if each column should be in 0-1 range (borders
    included)

    Returns
    -------
    dict.
    Dictionary containing various evaluation metrics.

    """
    # Get matrix representations of training and test data as well as their
    # labels
    x_train, y_train = get_x_and_y(train, features, label, normalize)
    x_test, y_test = get_x_and_y(test, features, label, normalize)

    print x_train.shape
    print len(y_train)
    clf.fit(x_train, y_train)
    # Predict labels of unknown instances
    y_pred = clf.predict(x_test)
    m1 = metrics.explained_variance_score(y_test, y_pred,
                                          multioutput="uniform_average")
    m2 = metrics.mean_absolute_error(y_test, y_pred,
                                     multioutput="uniform_average")
    m3 = metrics.mean_squared_error(y_test, y_pred,
                                    multioutput="uniform_average")
    m4 = metrics.median_absolute_error(y_test, y_pred)
    m5 = metrics.r2_score(y_test, y_pred, multioutput="uniform_average")
    # https://stats.stackexchange.com/questions/253892/regression-what-does-the-median-absolute-error-metric-say-about-the-models
    return {"explained_variance": m1,
            "mean_absolute_error": m2,
            "mean_squared_error": m3,
            "median_absolute_error": m4,
            "r2": m5,
            "root_mean_squared_error": math.sqrt(m3)
            }


def split_training_test(df, train_ratio, seed):
    """
    Splits dataset into a training and test set. Note that the tweets
    are randomly assigned to either of the two.

    Parameters
    df: pandas.DataFrame - dataset.
    train_ratio: float - between 0 and 1 indicates how many percent of the
    tweets should be used for training. The rest will be used for testing.
    seed: int - seed for PRNG to ensure reproducibility.

    Returns
    -------
    pandas.DataFrame, pandas.DataFrame, pandas.DataFrame.
    Original (shuffled) dataset, training and test set.

    """
    # How many tweets should be used for training?
    training_tweets = int(df.shape[0] * train_ratio)
    # print "#training examples", training_tweets
    # Shuffle dataset
    df.sample(frac=1, random_state=seed)
    # Split it
    train = df.iloc[:training_tweets, :].copy()
    test = df.iloc[training_tweets:, :].copy()

    return df, train, test


def build_classifier(src, clf, train_ratio, glove_src, use_tfidf,
                     label, features, seed, normalize):
    """
    Builds a classifier from scratch, i.e. preprocesses tweets, extracts
    features, trains and evaluates the classifier.

    Parameters
    ----------
    src: str - path to the csv dataset.
    clf: sklearn classifier - instantiated classifier to be trained.
    train_ratio: float - between 0 and 1 indicates how many percent of the
    tweets should be used for training. The rest will be used for testing.
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's an empty string, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly.
    label: str - column name in <train> and <test> for the column holding the
    labels.
    features: list of str - names of the features to be used for building the
    classifier.
    seed: int - used to initialize PRNG to ensure reproducible results.
    normalize: bool - True if each column should be in 0-1 range (borders
    included)

    """
    use_glove = False
    if len(glove_src) > 0:
        use_glove = True
    # Read dataset
    df = read_as_df(src)
    # Preprocess tweets
    df = clean_tweets(df)
    # Split into training and test set
    df, train, test = split_training_test(df, train_ratio, seed)
    # Transform training and test set separately into matrix representation
    # because for some features, e.g. BoW, the test set uses the data from the
    # training set
    train = add_features(train, is_training_set=True, glove_src=glove_src,
                         use_tfidf=use_tfidf)
    test = add_features(test, is_training_set=False, glove_src=glove_src,
                        use_tfidf=use_tfidf)
    results = train_and_evaluate(train, test, clf, features, label, normalize)
    print "Results"
    print results


def build_classifier_cv(src, folds, clf, train_ratio, glove_src,
                        use_tfidf, label, features, seed, normalize):
    """
    Builds a classifier from scratch, i.e. preprocesses tweets, extracts
    features, trains and evaluates the classifier. Uses cross-validation for
    estimating the classifier's performance.

    Parameters
    ----------
    src: str - path to the csv dataset.
    folds: int - number of folds in cross-validation.
    clf: sklearn classifier - instantiated classifier to be trained.
    train_ratio: float - between 0 and 1 indicates how many percent of the
    tweets should be used for training. The rest will be used for testing. This
    parameter is required for building the final model.
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's an empty string, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly.
    label: str - column name in <train> and <test> for the column holding the
    labels.
    features: list of str - names of the features to be used for building the
    classifier.
    seed: int - used to initialize PRNG to ensure reproducible results.
    normalize: bool - True if each column should be in 0-1 range (borders
    included)

    Returns
    -------
    dict.
    Resulting scores macro-averaged over the folds.

    """
    # Create a copy for building the final model
    final_clf = copy.deepcopy(clf)
    # Store evaluation results of cross-validation
    total = {
        "explained_variance": [],
        "mean_absolute_error": [],
        "mean_squared_error": [],
        "median_absolute_error": [],
        "r2": [],
        "root_mean_squared_error": []
    }
    # Read dataset
    df = read_as_df(src)
    # Preprocess tweets
    df = clean_tweets(df)
    # Shuffle dataset
    df.sample(frac=1, random_state=seed)
    kf = KFold(n_splits=folds, random_state=None, shuffle=False)
    # Number of tweets
    tweets = df.shape[0]
    # CV
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(range(tweets))):
        # Create a copy of the features, because otherwise BOW_PREFIX,
        # KEY_PREFIX, GLOVE_PREFIX are removed from the features in the
        # first run and the list of features from the previous fold is reused
        features_ = copy.deepcopy(features)
        # print "FOLD:", fold_idx+1
        # print "-------"
        # train_index contains indices of the rows used for training
        # Split into training and test set
        train = df.iloc[train_idx, :].copy()
        test = df.iloc[test_idx, :].copy()
        # print "train:", train.shape
        # print "test:", test.shape
        # Transform training and test set separately into matrix representation
        # because for some features, e.g. BoW, the test set uses the data from
        # the training set
        train = add_features(train, is_training_set=True, glove_src=glove_src,
                             use_tfidf=use_tfidf)
        test = add_features(test, is_training_set=False, glove_src=glove_src,
                            use_tfidf=use_tfidf)

        results = train_and_evaluate(train, test, clf, features_, label,
                                     normalize)
        # print "Results"
        # print results
        # print "\n"

        # Store results
        for metric in total:
            total[metric].append(results[metric])
    avg_results = {}
    print "Average results"
    print "----------------"
    # Average the metrics over all folds
    for metric, values in total.iteritems():
        avg_val = 1.0*sum(values) / len(values)
        print "{}: {}".format(metric, avg_val)
        avg_results[metric] = avg_val
    print "\n"

    # Create a copy of the features, because otherwise BOW_PREFIX,
    # KEY_PREFIX, GLOVE_PREFIX are removed from the features in the
    # first run and the list of features from the previous fold is reused
    features_ = copy.deepcopy(features)

    # Learn final model on whole training set
    build_classifier(src, final_clf, train_ratio, glove_src, use_tfidf,
                     label, features_, seed, normalize)
    return avg_results


##################################
# Preprocessing of glove vectors #
##################################
def store_glove_vectors_for_dataset(src, ds_src, dst):
    """
    Since the pre-trained glove vectors are large, it's time-consuming to read
    them each time. Thus, we create a subset (namely keeping only word vectors
    that exist in the dataset) and store it.
    Stores needed glove word vectors in the same directory as <src>.

    Parameters
    ---------
    src: str - path to pre-trained glove vectors.
    ds_src: str - path to dataset with tweets.
    dst: str - path where resulting file is stored.

    """
    w2v_src = convert_glove_to_word2vec_format(src)
    print "word2vec path", w2v_src
    model = gensim.models.KeyedVectors.load_word2vec_format(w2v_src,
                                                            binary=False)
    # print "word2vec vector for 'in':", model["in"]
    print "store results in", dst
    # Read dataset
    df = read_as_df(ds_src)
    # Preprocess tweets
    df = clean_tweets(df)
    tweets = df[PREPROCESSED].tolist()
    # {word: [high dimensional vector values in a list]}
    word_vectors = {}
    with codecs.open(dst, "w", encoding="utf-8") as f:
        for text in tweets:
            words = text.split()
            for word in words:
                word = word.lower()
                # Check if word exists in model
                if word in model.wv.vocab:
                    if word not in word_vectors:
                        word_vectors[word] = model[word].tolist()
        # https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
        data = json.dumps(word_vectors, encoding="utf-8", indent=4,
                          ensure_ascii=False)
        f.writelines(unicode(data))


def convert_glove_to_word2vec_format(src):
    """
    Test glove vector representation which preserves semantics of words, s.t.
    words with similar semantics have similar vectors.
    Only performs the conversion if the word2vec representation of <src> doesn't
    exist yet.

    Parameters
    ---------
    src: str - path to pre-trained glove vectors.

    Returns
    -------
    str.
    Path to the input file in word2vector format.

    """
    if not os.path.isfile(src):
        raise ValueError("Download the Glove vector files first!")

    glove_name = ntpath.basename(src)
    print "glove file name", glove_name
    w2v_name = glove_name.replace("glove", "word2vector")
    print "word2vec name", w2v_name
    f_dir = os.path.dirname(src)
    dst = os.path.join(f_dir, w2v_name)
    print "dst", dst
    if not os.path.isfile(dst):
        _convert_glove_to_word2vec_format(src, dst)
    return dst


def _convert_glove_to_word2vec_format(src, dst):
    """Converts file with glove vectors into word2vec format that is readable
    by gensim."""
    from gensim.scripts import glove2word2vec
    glove2word2vec.glove2word2vec(src, dst)


######################
# Actual experiments #
######################
# https://stackoverflow.com/questions/1482308/whats-a-good-way-to-combinate-through-a-set
def powerset(features):
    """
       Creates all possible feature subsets.

       Parameters
       ----------
       features: list of str - features to be used for building the classifier.

   """
    x = len(features)
    masks = [1 << i for i in range(x)]
    # No empty set is created
    for i in xrange(1, 1 << x):
        yield [ss for mask, ss in zip(masks, features) if i & mask]


def run_experiment(src, folds, train_ratio, glove_paths, classifiers,
                   classifier_names, dst_dir, label, features, seed):
    """
    Automatically perform all experiments and store results in files.
    Create 1 output file per classifier.

    Use https://stackoverflow.com/questions/1894269/convert-string-representation-of-list-to-list-in-python
    to read the results.

    File format per classifier:
    for each glove representation:
        settings: dict
        classifier_name: str
        classifier_params: dict
        #########
            for each glove weight (uniform or TF-IDF)
                glove_weight: str (either "tf-idf" or "uniform")
                **********
                for each feature subset
                used_features: list
                results: dict
                <new_line before writing results of next subset>
            ++++++++++
        ----------

    Parameters
    ----------
    src: str - path to the csv dataset.
    folds: int - number of folds in cross-validation.
    classifiers: list of sklearn classifier - instantiated classifiers to be
    trained.
    classifier_names: list of str - names of <classifiers> - must be in the
    same order
    train_ratio: float - between 0 and 1 indicates how many percent of the
    tweets should be used for training. The rest will be used for testing.
    dst_dir: str - directory in which the results will be stored.
    label: str - column name in <train> and <test> for the column holding the
    labels.
    features: list of str - names of the features to be used for building the
    classifier.
    seed: int - used to initialize PRNG to ensure reproducible results.

    """
    # Each classifier must have its own name
    assert(len(classifiers) == len(classifier_names))
    # For each classifier
    for clf, clf_name in zip(classifiers, classifier_names):
        # Get name of dataset
        ds_name = ntpath.basename(src).split(".")[0]
        fname = "{}_{}.txt".format(ds_name, clf_name)
        dst = os.path.join(dst_dir, fname)
        with open(dst, "wb") as f:
            # For each glove vector
            for glove_path in glove_paths:
                # Store experiment settings
                settings = {
                    "folds": folds,
                    "train_ratio": train_ratio,
                    "dataset": ds_name,
                    "glove_path": glove_path
                }
                f.write("settings:{}\n".format(repr(settings)))
                # Store classifier info
                f.write("classifier_name:{}\n".format(repr(clf_name)))
                # Store classifier parameters
                f.write("classifier_params:{}\n"
                        .format(repr(clf.get_params(deep=True))))
                f.write("##########\n")
                # Weigh glove vectors according to TF-IDF or uniformly
                for glove_weight in [True, False]:
                    print "\nglove_weight: {}\n".format(glove_weight)
                    if glove_weight:
                        weight = "tf-idf"
                    else:
                        weight = "uniform"
                    f.write("glove_weight:{}\n".format(repr(weight)))
                    f.write("**********\n")
                    # For all feature subsets
                    for fs in powerset(features):
                        print "subset:", fs
                        # Store feature info
                        f.write("used_features:{}\n".format(repr(fs)))

                        # Store results
                        results = \
                            build_classifier_cv(src, folds, clf, train_ratio,
                                                glove_path, glove_weight,
                                                label, fs, seed)
                        f.write("results:{}\n\n".format(repr(results)))
                        # break
                    f.write("++++++++++\n")
                    # break
                f.write("----------\n")
                # break


###############################
# Hyperparameter optimization #
###############################
def objective_gbr(params):
    global reg
    max_depth, learning_rate, max_features, min_samples_split, \
        min_samples_leaf = params

    reg.set_params(max_depth=max_depth,
                   learning_rate=learning_rate,
                   max_features=max_features,
                   min_samples_split=min_samples_split,
                   min_samples_leaf=min_samples_leaf)

    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
                                    scoring="neg_mean_squared_error"))


def objective_ridge(params):
    global reg
    alpha, tol = params

    reg.set_params(alpha=alpha,
                   tol=tol)

    return -np.mean(cross_val_score(reg, X, y, cv=5, n_jobs=-1,
                                    scoring="neg_mean_squared_error"))


def find_optimal_params_gbr(src, glove_src, use_tfidf, features, label, seed,
                            normalize):
    """
    Optimizes hyperparameters of gradient boosting regressor.

    Parameters
    ----------
    src: str - path to the csv dataset.
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's an empty string, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly.
    label: str - column name in <train> and <test> for the column holding the
    labels.
    features: list of str - names of the features to be used for building the
    classifier.
    seed: int - used to initialize PRNG to ensure reproducible results.
    normalize: bool - True if each column should be in 0-1 range (borders
    included)

    """
    global X
    global y
    features_ = copy.deepcopy(features)
    # Read dataset
    df = read_as_df(src)
    # Preprocess tweets
    df = clean_tweets(df)
    # Transform data into matrix representation
    train = add_features(df, is_training_set=True,
                         glove_src=glove_src,
                         use_tfidf=use_tfidf)
    X, y = get_x_and_y(train, features_, label, normalize)
    n_features = X.shape[1]
    space = [(0, 5),                           # max_depth
             (10**-5, 10**0, "log-uniform"),   # learning_rate
             (1, n_features),                  # max_features
             (2, 100),                         # min_samples_split
             (1, 100)                          # min_samples_leaf
             ]

    res_gp = gp_minimize(objective_gbr, space, n_calls=100, random_state=seed,
                         n_jobs=-1)
    print "Best score=%.4f" % res_gp.fun
    print("""Best parameters:
    - max_depth=%d
    - learning_rate=%.6f
    - max_features=%d
    - min_samples_split=%d
    - min_samples_leaf=%d""" % (res_gp.x[0], res_gp.x[1],
                                res_gp.x[2], res_gp.x[3],
                                res_gp.x[4]))


def find_optimal_params_ridge(src, glove_src, use_tfidf, features, label, seed,
                              normalize):
    """
    Optimizes hyperparameters of a ridge regressor.

    Parameters
    ----------
    src: str - path to the csv dataset.
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's an empty string, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly.
    label: str - column name in <train> and <test> for the column holding the
    labels.
    features: list of str - names of the features to be used for building the
    classifier.
    seed: int - used to initialize PRNG to ensure reproducible results.
    normalize: bool - True if each column should be in 0-1 range (borders
    included)

    """
    global X
    global y
    features_ = copy.deepcopy(features)
    # Read dataset
    df = read_as_df(src)
    # Preprocess tweets
    df = clean_tweets(df)
    # Transform data into matrix representation
    train = add_features(df, is_training_set=True,
                         glove_src=glove_src,
                         use_tfidf=use_tfidf)
    X, y = get_x_and_y(train, features_, label, normalize)
    space = [(1, 5),    # a
             (10 ** -5, 10 ** 0),  # tol
             ]

    res_gp = gp_minimize(objective_ridge, space, n_calls=100, random_state=seed,
                         n_jobs=-1)
    print "Best score=%.4f" % res_gp.fun
    print("""Best parameters:
    - alpha%d
    - tol=%.6f""" % (res_gp.x[0], res_gp.x[1]))


#####################
# Utility functions #
#####################
def to_arff(df, features_, label, dst):
    """
    Converts a dataframe into an arff file for Weka by considering specific
    features.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.
    features_: list of str - names of the features (=columns) in <df>.
    label: str - name of the column storing the class label in <df>.
    dst: str - path where arff file will be stored.

    """
    df.fillna("?")
    with open(dst, "wb") as f:
        feature_names = get_feature_names(df, features_)
        # Normalize in Weka if desired, not here
        matrix, labels = get_x_and_y(df, features_, label, False)
        print "dataset size", matrix.shape
        # Dataset name
        f.write("@RELATION Twitter\n\n")
        # Header
        # First attribute is tweet ID!
        f.write("@ATTRIBUTE ID STRING\n")
        for feature in feature_names:
            f.write("@ATTRIBUTE {} NUMERIC\n".format(feature))
        # Add class label as last attribute
        # Either regression or classification
        if label == LABEL_REG:
            f.write("@ATTRIBUTE class NUMERIC\n")
        else:
            # Skip last comma
            f.write("@ATTRIBUTE class {{{}}}\n".format(","
                                                       .join(AGREEMENT_LEVELS)))
        # Get the IDs from the data frame
        ids = list(df.index.values)
        # Actual data
        f.write("@DATA\n")
        rows = matrix.tolist()
        for i, row in enumerate(rows):
            # Separate values by commas
            line = ids[i] + "," + ",".join(str(e) for e in row)
            # Add label of instance
            line += ",{}".format(labels[i])
            f.write("{}\n".format(line))
        print "entries", len(line.split(","))


def export_to_weka(src, glove_src, use_tfidf, dst, features, label,
                   training=True):
    """
    Exports a dataset to Weka's arff format.

    Parameters
    ----------
    src: str - path to the dataset.
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's an empty string, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly.
    dst: str - path were arff file will be stored.
    features: list of str - names of the features to be used for building the
    classifier.
    label: str - column name in <train> and <test> for the column holding the
    labels.
    training: bool - True if BoW and TF-IDF vectorizers should be trained. Else
    the data is just transformed using the current ones.

    """
    # Read dataset
    df = read_as_df_gizem(src)
    # Preprocess tweets
    df = clean_tweets(df)
    # Transform data into matrix representation adding all features
    df = add_features(df, is_training_set=training, glove_src=glove_src,
                      use_tfidf=use_tfidf)
    to_arff(df, features, label, dst)


def export_to_weka_gizem(src, src_pred_rel, src_pred_sen, glove_src, use_tfidf,
                         dst, features, label, training=True):
    """
    Exports a dataset to Weka's arff format. Same as export_to_weka(), but
    additionally adds Gizem's extracted features.

    Parameters
    ----------
    src: str - path to the dataset.
    src_pred_rel: str - path to Weka's predictions for relevance.
    src_pred_rel: str - path to Weka's predictions for sentiment.
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's an empty string, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly.
    dst: str - path were arff file will be stored.
    features: list of str - names of the features to be used for building the
    classifier.
    label: str - column name in <train> and <test> for the column holding the
    labels.
    training: bool - True if BoW and TF-IDF vectorizers should be trained. Else
    the data is just transformed using the current ones.

    """
    # Read dataset
    df = read_as_df_gizem(src, src_pred_rel, src_pred_sen)
    # Preprocess tweets
    df = clean_tweets(df)
    # Transform data into matrix representation adding all features
    df = add_features(df, is_training_set=training, glove_src=glove_src,
                      use_tfidf=use_tfidf)
    to_arff(df, features, label, dst)


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Directory in which the crowdsourced dataset is stored
    DS_DIR = os.path.join(base_dir, "results", "dataset_twitter")
    # Directory in which the full TREC dataset is stored
    FULL_DIR = os.path.join(base_dir, "results", "dataset_twitter_full")
    # Directory in which the glove vectors existing in the Twutter dataset
    # will be stored
    TWITTER_DIR = os.path.join(base_dir, "results", "glove_twitter")
    # Path to pretrained glove vectors
    # Downloaded from here:
    # https://github.com/stanfordnlp/GloVe
    # 300 dimensional vectors, uncased, 42B tokens, 2.2 Mio vocabulary from
    # Wikipedia
    GLOVE_VEC_WIKI = os.path.join(base_dir, "results", "glove",
                                  "glove.42B.300d.txt")
    # 200 dimensional vectors, uncased, 27B tokens, 2B tweets,
    # 1.2 Mio vocabulary from Twitter
    GLOVE_VEC_TWITTER = os.path.join(base_dir, "results", "glove",
                                     "glove.twitter.27B",
                                     "glove.twitter.27B.200d.txt")
    # Will only exist after running store_glove_vectors_for_dataset() once
    # Path to glove vectors existing in our dataset in w2v format.
    # Trained on Wikipedia
    GLOVE_W2V_VEC_WIKI = os.path.join(TWITTER_DIR, "glove.42B.300d_glove_for_twitterc_watson_rosette_min_annos_3_cleaned.json")
    # Trained on Twitter
    GLOVE_W2V_VEC_TWITTER = os.path.join(TWITTER_DIR, "glove.twitter.27B.200d_glove_for_twitter_watson_rosette_min_annos_3_cleaned.json")
    # Directory in which the classifier results will be stored
    CLF_DIR = os.path.join(base_dir, "results", "classifier_results")
    # Directory in which the converted .arff files are stored
    ARFF_DIR = os.path.join(base_dir, "results", "arff_files")
    # Directory in which Weka predictions for relevance of the instances are
    # stored
    REL_PRED_DIR = os.path.join(base_dir, "results", "weka_predictions",
                                "sentiment_relevance_predictions_twitter")
    # Directory in which Weka predictions for relevance of the instances are
    # stored
    SEN_PRED_DIR = os.path.join(base_dir, "results/weka_predictions",
                                "sentiment_sentiment_predictions_twitter")
    if not os.path.exists(CLF_DIR):
        os.makedirs(CLF_DIR)
    if not os.path.exists(ARFF_DIR):
        os.makedirs(ARFF_DIR)
    if not os.path.exists(TWITTER_DIR):
        os.makedirs(TWITTER_DIR)

    # Use "cleaned" version of dataset, i.e. discard all other annotation times
    # if a tweet was assigned the label "Irrelevant"
    cleaned = True
    # Minimum number of annotators who labeled each tweet
    min_annos = 3
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    # fname = "dataset_trec_relevance_watson_rosette_{}.json".format(agg)
    fname = "dataset_twitter_watson_rosette_{}.json".format(agg)
    src = os.path.join(DS_DIR, fname)
    if not os.path.isfile(src):
        raise IOError("Dataset doesn't exist - import it first!")
    # Use BoW features in regression model?
    use_bow = False
    # Folds used for cross-validation
    folds = 3
    # Percentage of dataset used for training
    train_ratio = 0.5
    # True if glove vectors should be weighted according to their TF-IDF scores
    use_tfidf = True
    # Make sure each column is in range [0...1]
    normalize = True

    # Run this function to extract only the glove vectors that exist in
    # our dataset -> then reading in GBs of text data isn't necessary and it's
    # much faster
    # if not os.path.isfile(GLOVE_W2V_VEC_TWITTER):
    #     store_glove_vectors_for_dataset(GLOVE_VEC_TWITTER, src,
    #                                     GLOVE_W2V_VEC_TWITTER)
    # if not os.path.isfile(GLOVE_W2V_VEC_WIKI):
    #     store_glove_vectors_for_dataset(GLOVE_VEC_WIKI, src, GLOVE_W2V_VEC_WIKI)

    # vecs = read_glove_vectors(GLOVE_W2V_VEC_TWITTER)
    #
    # texts = ["hello and name is stefan", "hi name is ece"]
    #
    # df = pd.DataFrame(texts, columns=[PREPROCESSED])
    # print df
    # print "tfidfs"
    # tfidf_weights = get_tfidf(df)
    # for text in texts:
    #     print "text", text
    #     res = compute_glove_vector_for_tweet(text, vecs, tfidf=tfidf_weights)

    clf1 = linear_model.Ridge(alpha=1, tol=1e-2, solver="auto",
                              normalize=True, random_state=SEED)
    clf2 = linear_model.LinearRegression(fit_intercept=True, normalize=True)
    # clf = linear_model.Lasso(alpha=0.5, fit_intercept=True,
    # normalize=True)
    # clf = RandomForestRegressor(criterion="mae")

    # Simple: evaluate classifier by splitting data into training and test set
    # build_classifier(src, clf1, train_ratio, GLOVE_W2V_VEC_TWITTER,
    #                  use_tfidf, LABEL, FEATURES, SEED, normalize)

    # Realistic: evaluate classifier using cross-validation
    # build_classifier_cv(src, folds, clf1, train_ratio,
    #                     GLOVE_W2V_VEC_TWITTER, use_tfidf, LABEL, FEATURES,
    #                     SEED, normalize)
    glove_paths = [GLOVE_W2V_VEC_TWITTER, GLOVE_W2V_VEC_WIKI]
    classifiers = [clf1, clf2]
    classifier_names = ["ridge", "linreg"]

    # run_experiment(src, folds, train_ratio, glove_paths, classifiers,
    #                classifier_names, CLF_DIR, LABEL, FEATURES, SEED,
    #                normalize)
    #

    global reg
    ##########################################
    # Optimized parameters of the classifier after running
    # find_optimal_params_gbr().
    # With normalize = False and True:
    ###########################
    # RMSE: 0.234
    # clf1 = GradientBoostingRegressor(n_estimators=50, random_state=SEED,
    #                                  min_samples_leaf=100,
    #                                  min_samples_split=100,
    #                                  learning_rate=0.123535,
    #                                  max_depth=1)
    # reg = GradientBoostingRegressor(n_estimators=50, random_state=SEED)
    # find_optimal_params_gbr(src, GLOVE_W2V_VEC_TWITTER, use_tfidf, FEATURES,
    #                         LABEL, SEED, normalize)

    ##########################################
    # Optimized parameters of the classifier after running
    # find_optimal_params_ridge().
    # With normalize = False:
    ###########################
    # RMSE: 0.233
    # clf1 = linear_model.Ridge(alpha=5, tol=0.453455, solver="auto",
    #                           normalize=True, random_state=SEED)
    # reg = linear_model.Ridge(alpha=1, tol=1e-2, solver="auto", normalize=True,
    #                          random_state=SEED)
    # find_optimal_params_ridge(src, GLOVE_W2V_VEC_TWITTER, use_tfidf, FEATURES,
    #                           LABEL, SEED, normalize)
    # print build_classifier_cv(src, folds, clf1, train_ratio,
    #                           GLOVE_W2V_VEC_TWITTER, use_tfidf, LABEL, FEATURES,
    #                           SEED, normalize)

    ###################################################################
    # Now perform Weka experiments with the crowdsourced TREC dataset #
    # (=Training set)                                                 #
    ###################################################################
    # fname = "dataset_trec_watson_rosette_{}.json".format(agg)
    # src_crowd = os.path.join(DS_DIR, fname)
    # if not os.path.isfile(src_crowd):
    #     raise IOError("Dataset doesn't exist - import it first!")
    # fname = "dataset_trec_full_watson_rosette_{}.json".format(agg)
    # src_full = os.path.join(DS_DIR, fname)
    # if not os.path.isfile(src_crowd):
    #     raise IOError("Dataset doesn't exist - import it first!")
    # # a) Formulate problem as a regression task
    # # Create dataset once using Wikipedia glove vectors
    # fname = "dataset_trec_wikipedia_regression.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # export_to_weka(src, GLOVE_W2V_VEC_WIKI, use_tfidf, dst, FEATURES, LABEL_REG)
    #
    # # And create dataset once using Twitter glove vectors
    # fname = "dataset_trec_twitter_regression.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # export_to_weka(src, GLOVE_W2V_VEC_TWITTER, use_tfidf, dst, FEATURES,
    #                LABEL_REG)

    # # b) Formulate problem as a classification task
    # # Create dataset once using Wikipedia glove vectors
    # pname = "gizem_relevance_logistic.csv"
    # pred_dst = os.path.join(PRED_DIR, pname)
    # fname = "dataset_trec_relevance_final_wikipedia_classification.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # export_to_weka_gizem(src, pred_dst, GLOVE_W2V_VEC_WIKI, use_tfidf, dst,
    #                      FEATURES, LABEL_CLA, training=True)
    #
    # # And create dataset once using Twitter glove vectors
    # fname = "dataset_trec_relevance_final_twitter_classification.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # export_to_weka_gizem(src, pred_dst, GLOVE_W2V_VEC_TWITTER, use_tfidf, dst,
    #                      FEATURES, LABEL_CLA, training=True)

    ###########################################################
    # Now perform Weka experiments with the full TREC dataset (test set) #
    ###########################################################
    # But ignore all those crowdsourced TREC tweets
    # a) Formulate problem as a regression task
    # Create dataset once using Wikipedia glove vectors
    # fname = "dataset_trec_full_wikipedia_regression.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # export_to_weka(src, GLOVE_W2V_VEC_WIKI, use_tfidf, dst, FEATURES, LABEL_REG)
    #
    # # And create dataset once using Twitter glove vectors
    # fname = "dataset_trec_full_twitter_regression.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # export_to_weka(src, GLOVE_W2V_VEC_TWITTER, use_tfidf, dst, FEATURES,
    #                LABEL_REG)

    # # b) Formulate problem as a classification task
    # # Create dataset once using Wikipedia glove vectors
    # fname = "dataset_trec_full_wikipedia_classification.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # export_to_weka(src, GLOVE_W2V_VEC_WIKI, use_tfidf, dst, FEATURES, LABEL_CLA)
    #
    # # And create dataset once using Twitter glove vectors
    # fname = "dataset_trec_full_twitter_classification.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # export_to_weka(src, GLOVE_W2V_VEC_TWITTER, use_tfidf, dst, FEATURES,
    #                LABEL_CLA)

    #########################################################################
    # Build training (Gizem's crowdsourced TREC dataset) and test set (Full #
    # TREC dataset)                                                         #
    #########################################################################
    # b) Formulate problem as a classification task
    # Create dataset once using Wikipedia glove vectors
    # Training set
    # pname = "add_classifier_twitter_sentiment.csv"
    # pred_dst = os.path.join(PRED_DIR, pname)
    # fname = "dataset_twitter_agreement_final_wikipedia_classification.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # export_to_weka_gizem(src, pred_dst, GLOVE_W2V_VEC_WIKI, use_tfidf, dst,
    #                      FEATURES, LABEL_CLA, training=True)
    #
    # # Test set
    # pname = "add_classifier_twitter_full_sentiment.csv"
    # pred_dst = os.path.join(PRED_DIR, pname)
    # fname = "dataset_trec_full_relevance_final_wikipedia_watson_rosette_{}" \
    #         ".json".format(agg)
    # src = os.path.join(FULL_DIR, fname)
    # fname = "dataset_twitter_full_agreement_final_wikipedia_classification.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # export_to_weka_gizem(src, pred_dst, GLOVE_W2V_VEC_WIKI, use_tfidf, dst,
    #                      FEATURES, LABEL_CLA, training=False)

    # And create dataset once using Twitter glove vectors
    # Training set
    pname = "sentiment_relevance_twitter_training.csv"
    rel_pred_src = os.path.join(REL_PRED_DIR, pname)
    pname = "sentiment_sentiment_twitter_training.csv"
    sen_pred_src = os.path.join(SEN_PRED_DIR, pname)
    fname = "dataset_twitter_agreement_final_twitter_classification.arff"
    dst = os.path.join(ARFF_DIR, fname)
    export_to_weka_gizem(src, rel_pred_src, sen_pred_src, GLOVE_W2V_VEC_TWITTER,
                         use_tfidf, dst, FEATURES, LABEL_CLA, training=True)

    # Test set
    #############################################################
    # IMPORTANT: whenever test set should be trained, training set MUST BE
    # TRAINED FIRST because otherweise <TFIDF_WEIGHTS> is None
    #############################################################
    # Trained on Twitter
    GLOVE_W2V_VEC_TWITTER = os.path.join(TWITTER_DIR,
                                         "glove.twitter.27B.200d_glove_for_twitter_full_watson_rosette_min_annos_3_cleaned.json")
    pname = "sentiment_relevance_twitter_test.csv"
    rel_pred_src = os.path.join(REL_PRED_DIR, pname)
    pname = "sentiment_sentiment_twitter_test.csv"
    sen_pred_src = os.path.join(SEN_PRED_DIR, pname)
    fname = "dataset_twitter_full_agreement_final_twitter_classification.arff"
    dst = os.path.join(ARFF_DIR, fname)
    fname = "dataset_twitter_full_watson_rosette_{}.json".format(agg)
    src = os.path.join(FULL_DIR, fname)
    export_to_weka_gizem(src, rel_pred_src, sen_pred_src, GLOVE_W2V_VEC_TWITTER,
                         use_tfidf, dst, FEATURES, LABEL_CLA, training=False)


