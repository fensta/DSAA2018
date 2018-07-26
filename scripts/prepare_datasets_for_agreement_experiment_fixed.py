"""
Extract various features from the dataset to build a numeric representation of
the texts using various features (see add_features() for an exhaustive list).

It's used for investigating if the agreement classifier becomes better after
the 2nd round. -> use HIGH + MEDIUM + LOW + training data and do 10-fold CV.
It's also used to check if performance differences occur depending on using 4
or 8 labels. -> use only LOW dataset with 4 or 8 labels and 10-fold CV.

Contrary to preprare_datasets_for_agreement_experiment.py, here agreement labels
are just "high" and "low" using Myra's proposed definition: "if majority_label
has <= 50% of all votes, it's low agreement, else high". In the other script
3 levels are used according to agreement scores.

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
import unicodecsv as csv
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import gensim
from sklearn import preprocessing
from pyjarowinkler import distance

from create_dataset_twitter import read_json


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
SENTIMENT_LABEL = "sentiment_label"
RELEVANCE_LABEL = "relevance_label"

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
AGREEMENT_LEVELS = ["low", "high"]

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

GROUND_TRUTH = {
    "Irrelevant": "Irrelevant",
    "Relevant": "Relevant",
    "Factual": "Neutral",
    "Positive": "Positive",
    "Negative": "Negative"
}

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
# Features I extracted - keep them separate because merge_to_df() relies on that
# Ignore all _PREFIX features because they're just indicators to add some
# features. But this is done using <FEATURES> and not <MY_FEATURES>
MY_FEATURES = [SENTIMENT, RATIO_STOPWORDS, #KEY_PREFIX,
            # COSTS,  # only available in training set
            LENGTH, COSINE, RA_PROP,
            # HASH, # every tweet has hashtag
            # BOW_PREFIX,
            # MENTION,  # almost no tweet has one
            # URL, # no tweet contains URLs that's why they were chosen
            NER_NUMBER, DIVERSITY, EXTRA_TERMS, MISSING_TERMS,
            OVERLAPPING_TERMS, WEIRD, LCS, DOT, JARO_WINKLER]
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
            OVERLAPPING_TERMS, WEIRD, LCS, DOT, JARO_WINKLER]
# FEATURES.extend(GIZEMS_NAMES)
# Same as <reg>, but the instances
X = None
# Labels of instances
y = None

# Relevance of our Trump tweets with respect to our "query" (=some keywords)
CUSTOM_QUERY = "donald trump hillary clinton political election discussion " \
               "campaign"


######################
# Feature extraction #
######################
def add_features(df, tfidf_dst, is_training_set=True, glove_src="",
                 use_tfidf=False):
    """
    Adds columns to the data frame that contains multiple features.

    Parameters
    ----------
    df: pandas.DataFrame - dataset.
    tfidf_dst: str - path where TF-IDF matrix will be stored.
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

    # TODO: BOW is disabled to save RAM
    if is_training_set:
        # df = add_bow_for_training_set(df)
        if use_tfidf:
            TFIDF_WEIGHTS = get_tfidf(df)
            with codecs.open(tfidf_dst, "w") as fp:
                data = json.dumps(TFIDF_WEIGHTS, encoding="utf-8",
                                  ensure_ascii=False)
                fp.writelines(unicode(data))
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
    # Use original text to find @ and # and so on
    text_words = row[TEXT].split()

    # 2. Extract features related to text
    for word in text_words:
        if word.startswith("#"):
            has_hashtag = True
        if word.startswith("http"):
            has_url = True
        if word.startswith("@"):
            has_hashtag = True

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
    # Tell Python we want to update the existing global variable instead of
    # creating a new local variable with the same name
    # global TFIDF_VECTORIZER
    vectorizer = TfidfVectorizer()
    # Learn weights
    X = vectorizer.fit_transform(df[PREPROCESSED])
    # Store vectorizer so that test set can be transformed later on
    # TFIDF_VECTORIZER = vectorizer

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


def remove_stopwords(text):
    """Removes stopwords from the string"""
    tokenized = text.split()
    filtered_words = [word for word in tokenized if
                      word not in stopwords.words('english')]
    text = " ".join(filtered_words)
    return text


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
def read_as_df(src, crowd_tids):
    """
    Reads in the dataset into a data frame object.

    Parameters
    ----------
    src: str - path to dataset.
    crowd_tids: list - list of tweet IDs used in one of the 3 crowdsourcing
    datasets.

    Returns
    -------
    pandas.DataFrame.
    Column names: text, agree_score, votes (= #annotators who labeled it),
    labeling_cost

    """
    tweets = read_json(src)
    tids = []
    texts = []
    labels = []
    sentiments = []
    watson_ners = []
    queries = []
    rosette_ners = []
    keywords = []
    pos_tags = []
    agree_labels = []
    for tid in crowd_tids:
        tids.append(tid)
        t = tweets[tid]
        agree_labels.append(t[LABEL_CLA])
        texts.append(t[TEXT])
        # Only true if it's a training set
        rosette_ners.append(t[NER_ROSETTE])
        queries.append(CUSTOM_QUERY)
        watson_ners.append(t[NER_WATSON])
        pos_tags.append(t[POS_TAG])
        sentiments.append(t[SENTIMENT])
        keywords.append(t[KEYWORDS])
    data = {
        TEXT: texts,
        LABEL_CLA: agree_labels,
        SENTIMENT: sentiments,
        NER_WATSON: watson_ners,
        NER_ROSETTE: rosette_ners,
        KEYWORDS: keywords,
        QUERY: queries,
        POS_TAG: pos_tags
    }

    df = pd.DataFrame(index=tids, data=data)
    return df


def read_as_df_gizem(src, crowd_tids):
    """
    Reads in the dataset into a data frame object. Same as read_as_df(), but
    adds Gizem's extracted features as well

    Parameters
    ----------
    src: str - path to dataset.
    crowd_tids: list - list of tweet IDs used in one of the 3 crowdsourcing
    datasets.

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
    queries = []
    pos_tags = []
    labels = []
    # List of lists. The i-th inner list represents her i-th feature according
    # to <GIZEMS_NAMES>.
    gizems_features = [[] for _ in xrange(len(GIZEMS_NAMES))]
    # We only need the tweets from the experiment
    for tid in crowd_tids:
        tids.append(tid)
        t = tweets[tid]
        labels.append(t[LABEL_CLA])
        texts.append(t[TEXT])
        rosette_ners.append(t[NER_ROSETTE])
        watson_ners.append(t[NER_WATSON])
        queries.append(CUSTOM_QUERY)
        pos_tags.append(t[POS_TAG])
        sentiments.append(t[SENTIMENT])
        keywords.append(t[KEYWORDS])
        # Add Gizem's features
        for idx, name in enumerate(GIZEMS_NAMES):
            gizems_features[idx].append(t[name])
    data = {
        TEXT: texts,
        LABEL_CLA: labels,
        SENTIMENT: sentiments,
        NER_WATSON: watson_ners,
        NER_ROSETTE: rosette_ners,
        KEYWORDS: keywords,
        QUERY: queries,
        POS_TAG: pos_tags
    }
    for idx, f in enumerate(gizems_features):
        feature_name = GIZEMS_NAMES[idx]
        data[feature_name] = gizems_features[idx]
    df = pd.DataFrame(index=tids, data=data)
    return df


def merge_to_df(tweets, features):
    """
    Merges the existing features.

    Parameters
    ----------
    tweets: dict - dataset {tid: {"feature1": ..., "feature 2": "..."...}.
    features: dict - Gizem's extracted features {tid: {"f3": "...", ...}

    Returns
    -------
    pandas.DataFrame.
    Column names: text, agree_score, votes (= #annotators who labeled it),
    labeling_cost

    """
    # Get names of glove vector dimensions (glove_1, glove_2...) in dataset
    glove_names = []
    tid = tweets.keys()[0]
    for k in tweets[tid]:
        if k.startswith("glove_"):
            glove_names.append(k)
    tids = []
    texts = []
    sentiments = []
    watson_ners = []
    rosette_ners = []
    keywords = []
    queries = []
    pos_tags = []
    labels = []
    # Add my extracted features
    my_features = [[] for _ in MY_FEATURES]
    # List of glove dimensions used (glove_1, glove_2...)
    gloves = [[] for _ in glove_names]
    # List of lists. The i-th inner list represents her i-th feature according
    # to <GIZEMS_NAMES>.
    gizems_features = [[] for _ in xrange(len(GIZEMS_NAMES))]
    # We only need the tweets from the experiment
    for tid in tweets:
        tids.append(tid)
        t = tweets[tid]
        labels.append(t[LABEL_CLA])
        texts.append(t[TEXT])
        rosette_ners.append(t[NER_ROSETTE])
        watson_ners.append(t[NER_WATSON])
        queries.append(CUSTOM_QUERY)
        pos_tags.append(t[POS_TAG])
        sentiments.append(t[SENTIMENT])
        keywords.append(t[KEYWORDS])
        # Add Gizem's features
        for idx, name in enumerate(GIZEMS_NAMES):
            gizems_features[idx].append(features[tid][name])
        # Add glove dimensions
        for idx, k in enumerate(glove_names):
            gloves[idx].append(tweets[tid][k])
        # Add my features
        for idx, k in enumerate(MY_FEATURES):
            my_features[idx].append(tweets[tid][k])
    data = {
        TEXT: texts,
        LABEL_CLA: labels,
        SENTIMENT: sentiments,
        NER_WATSON: watson_ners,
        NER_ROSETTE: rosette_ners,
        KEYWORDS: keywords,
        QUERY: queries,
        POS_TAG: pos_tags
    }
    # Add Gizem's features
    for idx, f in enumerate(gizems_features):
        feature_name = GIZEMS_NAMES[idx]
        data[feature_name] = gizems_features[idx]
    # Add glove dimensions
    for idx, f in enumerate(gloves):
        feature_name = glove_names[idx]
        data[feature_name] = gloves[idx]
    # Add my features
    for idx, f in enumerate(my_features):
        feature_name = MY_FEATURES[idx]
        data[feature_name] = my_features[idx]
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
    df[PREPROCESSED_NO_STOPWORDS] = df[PREPROCESSED].apply(remove_stopwords)
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


##################################
# Preprocessing of glove vectors #
##################################
def store_glove_vectors_for_dataset(src, df, dst):
    """
    Since the pre-trained glove vectors are large, it's time-consuming to read
    them each time. Thus, we create a subset (namely keeping only word vectors
    that exist in the dataset) and store it.
    Stores needed glove word vectors in the same directory as <src>.

    Parameters
    ---------
    src: str - path to pre-trained glove vectors.
    df: pandas.DataFrame - dataset.
    dst: str - path where resulting file is stored.

    """
    w2v_src = convert_glove_to_word2vec_format(src)
    print "word2vec path", w2v_src
    model = gensim.models.KeyedVectors.load_word2vec_format(w2v_src,
                                                            binary=False)
    # print "word2vec vector for 'in':", model["in"]
    print "store results in", dst
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
    with open(dst, "wb") as f:
        print features_
        feature_names = get_feature_names(df, features_)
        print "COS?", COSINE in feature_names
        print df[COSINE]
        # Normalize in Weka if desired, not here
        matrix, labels = get_x_and_y(df, features_, label, False)
        print "dataset size", matrix.shape
        # Dataset name
        f.write("@RELATION Twitter\n\n")
        # Header
        # First attribute are IDs!
        f.write("@ATTRIBUTE ID STRING\n")
        for feature in feature_names:
            f.write("@ATTRIBUTE {} NUMERIC\n".format(feature))
        # Add class label
        f.write("@ATTRIBUTE class {{{}}}\n".format(",".join(AGREEMENT_LEVELS)))
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


def to_csv(df, features_, label, dst):
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
    with open(dst, "wb") as f:
        writer = csv.writer(f, encoding="utf-8", dialect="excel",
                            delimiter="\t")
        # Add tweet ID and class label to features
        header = ["tweetID"]
        feature_names = get_feature_names(df, features_)
        header.extend(feature_names)
        header.append(LABEL_CLA)
        # Write header
        writer.writerow(header)
        # Normalize in Weka if desired, not here
        matrix, labels = get_x_and_y(df, features_, label, False)
        print "dataset size", matrix.shape
        # Get the IDs from the data frame
        ids = list(df.index.values)
        rows = matrix.tolist()
        for i, row in enumerate(rows):
            # Add tweet ID to front
            l = [ids[i]]
            l.extend(row)
            # And label to end
            l.append(labels[i])
            writer.writerow(l)


def read_crowdsourcing_ds():
    """
    Reads in tweet IDs of all tweets used in the 3 crowdsourcing datasets
    for Gizem's experiment.

    Returns
    -------
    dict.
    {tid: {text: ""}}

    """
    crowd_dir = "/media/data/Workspaces/PythonWorkspace/phd/Analyze-Labeled-Dataset/www2018_results/crowdsourcing_datasets/"
    tids = {}
    for fn in os.listdir(crowd_dir):
        src = os.path.join(crowd_dir, fn)
        # Skip folders
        if os.path.isfile(src):
            with open(src, "rb") as f:
                reader = csv.reader(f)
                for idx, row in enumerate(reader):
                    # Skip header
                    if idx > 0:
                        tid, _, text = row
                        tids[tid] = {TEXT: text}
    return tids


def get_watson_data(src_dir, crowd_tids):
    """
    Extracts overall tweet sentiment, NERs (lowercase), and keywords
    (lowercase). Sentiment per keyword as well as sentiment per NER is ignored,
    because they're mostly 0, so it's pointless.

    Parameters
    ----------
    src_dir: str - directory in which the sentiment data is stored.
    crowd_tids: list - list of tweet IDs used in 1 of the 3 crowdsourcing
    experiments.

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
    for tid in crowd_tids:
        fname = "{}.json".format(tid)
        fpath = os.path.join(src_dir, fname)
        # Some tweet IDs are either from training or crowdsourcing set
        if os.path.isfile(fpath):
            with codecs.open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f, encoding="utf-8")
            # Extract data
            # a) Keywords (lowercase)
            keywords = []
            # Some tweets were written in an unsupported language, e.g. arabic,
            # so no keywords were extracted and hence the tweet will be discarded
            if "keywords" in data and "sentiment" in data and "entities" in data:
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
                    KEYWORDS: keywords,
                    SENTIMENT: sentiment,
                    NER_WATSON: entities
                }
    return tweets


def get_rosette_data(src_dir, crowd_tids):
    """
    Extracts POS-tags (without punctuation) and NERs (lowercase).
    Sentiment per NER and overall
    sentiment are ignored as they only provide labels, but no scores, plus
    NER sentiment is mainly neutral, i.e. insufficient context.

    Parameters
    ----------
    src_dir: str - directory in which the sentiment data is stored.
    crowd_tids: list - list of tweet IDs used in 1 of the 3 crowdsourcing
    experiments.

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
    for tid in crowd_tids:
        fname = "{}.json".format(tid)
        fpath = os.path.join(src_dir, fname)
        # Some tweet IDs are either from training or crowdsourcing set
        if os.path.isfile(fpath):
            with codecs.open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f, encoding="utf-8")
            # Extract data
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
                POS_TAG: pos_tags,
                NER_ROSETTE: entities
            }
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


def dict_to_df(tweets):
    """
    Converts a dictionary to a pandas.DataFrame using the keys of the dictionary
    as indices in the data frame.

    Parameters
    ----------
    tweets: dict - {tid: {key1: val1}}

    Returns
    -------
    pandas.DataFrame.

    """
    texts = []
    tids = []
    labels = []
    rosette_ners = []
    watson_ners = []
    queries = []
    pos_tags = []
    sentiments = []
    keywords = []
    sent_labels = []
    rel_labels = []
    for tid in tweets:
        tids.append(tid)
        t = tweets[tid]
        texts.append(t[TEXT])
        rosette_ners.append(t[NER_ROSETTE])
        watson_ners.append(t[NER_WATSON])
        queries.append(CUSTOM_QUERY)
        pos_tags.append(t[POS_TAG])
        sentiments.append(t[SENTIMENT])
        keywords.append(t[KEYWORDS])
        labels.append(t[LABEL_CLA])
        sent_labels.append(t[SENTIMENT_LABEL])
        rel_labels.append(t[RELEVANCE_LABEL])
    data = {
        TEXT: texts,
        LABEL_CLA: labels,
        SENTIMENT: sentiments,
        NER_WATSON: watson_ners,
        NER_ROSETTE: rosette_ners,
        KEYWORDS: keywords,
        QUERY: queries,
        POS_TAG: pos_tags,
        RELEVANCE_LABEL: rel_labels,
        SENTIMENT_LABEL: sent_labels
    }
    df = pd.DataFrame(index=tids, data=data)
    return df


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
            text = t[TEXT]
            # Remove all line breaks in a tweet
            t[TEXT] = text.replace('\n', ' ').replace('\r', '')
            # t[TEXT] = text.decode("utf-8")
            # Only exists for crowdsourced Twitter dataset, but not for the full
            # version
            if "labels" in t:
                del t["labels"]

        # https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
        data = json.dumps(tweets, encoding="utf-8", indent=4,
                          ensure_ascii=False)
        f.writelines(unicode(data))


def read_amt_csv(src, votes, votes_per_tweet):
    """
    Reads in a csv file downloaded from AMT that contains the crowdsourced
    labels.

    Parameters
    ----------
    src: str - path to csv file.
    votes: int - number of annotator labels to consider. We always take the
    first <votes> labels.
    votes_per_tweet: int - specify how often each tweet was labeled in total.

    Returns
    -------
    dict.
    {tid:
        {<LABEL_CLA>: ""  # low or high
        TEXT: "",
        <SENTIMENT_LABEL>: ""  # pos, neg, neutral, or empty if irrelevant
        <RELEVANCE_LABEL>: ""  # relevant or irrelevant
        }

    """
    labels = {}
    with open(src, "rb") as f:
        reader = csv.reader(f)
        tweet_labels = []
        # Number of labels collected for the current tweet
        used_labels = 0
        for idx, row in enumerate(reader):
            # Skip header
            if idx > 0:
                if used_labels < votes:
                    label = row[-1]
                    label = LABEL_MAPPING[label]
                    tweet_labels.append(label)
                    used_labels += 1
                # Every <votes_per_tweet> times we extracted all labels of a
                # tweet
                if idx % votes_per_tweet == 0 and used_labels == votes:
                    tid = row[-4]
                    text = row[-2]
                    # Compute agreement
                    distrib = Counter(tweet_labels)
                    majority, count = distrib.most_common()[0]
                    sent_label = ""
                    rel_label = "Irrelevant"
                    if majority != "Irrelevant":
                        sent_label = majority
                        rel_label = "Relevant"
                    # High agreement
                    label = AGREEMENT_LEVELS[1]
                    # Low agreement
                    if count <= 1.0 * votes / 2:
                        label = AGREEMENT_LEVELS[0]
                    labels[tid] = {
                        LABEL_CLA: label,
                        TEXT: text,
                        SENTIMENT_LABEL: sent_label,
                        RELEVANCE_LABEL: rel_label
                    }

                    # Reset variables for next tweet
                    used_labels = 0
                    tweet_labels = []
    return labels


def get_agreement_and_sentiment_labels_training(tweets):
    """
    Computes agreement of annotator labels per tweet in the training set.
    If majority vote is <= 50% of the tweets, it's low agreement else high.
    Also computes the majority label for relevance and sentiment according
    to the mapping below to match the crowdsourcing task.

    ###############################################################
    # Map training labels to crowdsourced datasets per annotator: #
    # Relevant + factual = Neutral                                #
    # Relevant + non-factual + positive = Positive                #
    # Relevant + non-factual + negative = Negative                #
    # Irrelevant = Irrelevant                                     #
    ###############################################################

    Parameters
    ----------
    tweets: dict - {tid: {f1: "...", "labels": [[labels of anno1],
    [labels of anno2]]}}

    Returns
    -------
    dict.
    {tid:
        {
            <LABEL_CLA>: agreement_label,
            <SENTIMENT_LABEL>: "..."    # pos, neg, neu or empty if irrelevant
            <RELEVANCE_LABEL>: "..."    # relevant or irrelevant
        }
    }

    """
    all_labels = {}
    # Map the labels according to the abovementioned scheme to the crowdsourcing
    # task
    for tid in tweets:
        labels = tweets[tid]["labels"]
        mapped_labels = []
        for anno_labels in labels:
            # Relevant
            if anno_labels[0] == "Relevant":
                # Non-factual
                if anno_labels[1] == "Non-factual":
                    # Positive or negative
                    label = GROUND_TRUTH[anno_labels[2]]
                # Neutral
                else:
                    label = GROUND_TRUTH[anno_labels[1]]
            # Irrelevant
            else:
                label = GROUND_TRUTH[anno_labels[0]]
            mapped_labels.append(label)
        mapped_labels = Counter(mapped_labels)
        # print "counts", mapped_labels

        # 1. Now determine majority label
        all_labels[tid] = {}
        majority, cnt = mapped_labels.most_common()[0]
        # print "majority", majority

        # 2. Set relevance and sentiment label
        if majority != "Irrelevant":
            all_labels[tid][RELEVANCE_LABEL] = GROUND_TRUTH["Relevant"]
            all_labels[tid][SENTIMENT_LABEL] = majority
        else:
            all_labels[tid][RELEVANCE_LABEL] = GROUND_TRUTH["Irrelevant"]
            all_labels[tid][SENTIMENT_LABEL] = ""

        # 3. Set agreement label
        votes = sum(mapped_labels.values())
        # Low agreement
        if 1.0*cnt <= votes/2:
            all_labels[tid][LABEL_CLA] = AGREEMENT_LEVELS[0]
        # High agreement
        else:
            all_labels[tid][LABEL_CLA] = AGREEMENT_LEVELS[1]
        # print "majority: {} votes: {} Agreement label: {}"\
        #     .format(cnt, votes, agree_labels[tid])
    return all_labels


def export_to_csv_crowd(
        low, medium, high, dst, csv_dir, tfidf_dir, rosette_dir,
        watson_dir,
        rosette_dir_train, watson_dir_train, glove_src, use_tfidf, features,
        first_n, first_n_low):
    """
    Creates combined dataset of the 3 crowdsourced datasets LOW, MEDIUM, HIGH.

    Parameters
    ----------
    low: str - path to LOW crowdsourced dataset.
    medium: str - path to MEDIUM crowdsourced dataset.
    high: str - path to HIGH crowdsourced dataset.
    dst: str - path where dataset will be stored.
    csv_dir: str - path to directory in which csv files will be stored.
    tfidf_dir: str - path to directory in which the tfidf_weights will be stored
    rosette_dir: str - path where the features extracted via Rosette are stored.
    watson_dir: str - path where the features extracted via Watson are stored.
    rosette_dir_train: str - path where the features extracted via Rosette are
    stored for the training set.
    watson_dir_train: str - path where the features extracted via Watson are
    stored for the training set
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's an empty string, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly.
    features: list of str - names of the features to be used for building the
    classifier.
    first_n: int - number of first labels to use from crowdsourcing datasets.
    first_n_low: int - number of first labels to use from crowdsourcing
    datasets for dataset LOW (because up to 8 labels are available).

    """
    low_tweets = read_amt_csv(low, first_n_low, 8)
    med_tweets = read_amt_csv(medium, first_n, 4)
    hig_tweets = read_amt_csv(high, first_n, 4)
    # Get tweet IDs of all tweets that were used in any of the 3 crowdsourcing
    # datasets, i.e. 3000, and the training set, in total 3.5k
    tweets = copy.deepcopy(low_tweets)
    tweets.update(med_tweets)
    tweets.update(hig_tweets)
    tids = tweets.keys()

    # ##################################
    # Use this to commented code to create datasets for Gizem
    # First run complete function to get tf-idf weights, then rerun it using
    # only this part to overwrite the .csv files (after sending them, rerun
    # the whole function!)
    # file_name = os.path.basename(dst).split(".")[0]
    # dst_gizem = os.path.join(csv_dir, "{}.csv".format(file_name))
    # with open(dst_gizem, "wb") as f:
    #     writer = csv.writer(f, delimiter="\t")
    #     for tid in tweets:
    #         writer.writerow([tid, tweets[tid][TEXT]])
    # #################################

    print "#total tweets", len(tweets)
    print "#tids", len(tids)
    # Merge Watson/Rosette features of training and crowdsourcing set
    watson = get_watson_data(watson_dir, tids)
    watson_train = get_watson_data(watson_dir_train, tids)
    watson.update(watson_train)
    rosette = get_rosette_data(rosette_dir, tids)
    rosette_train = get_rosette_data(rosette_dir_train, tids)
    rosette.update(rosette_train)
    tids = tweets.keys()

    # Run this function to extract only the glove vectors that exist in
    # our dataset -> then reading in GBs of text data isn't necessary and it's
    # much faster
    if not os.path.isfile(glove_src):
        print "watson", len(watson)
        print "rosette", len(rosette)
        merged = merge_dicts(tweets, watson, rosette)
        dst = "/media/data/Workspaces/PythonWorkspace/phd/Analyze-Labeled-" \
              "Dataset/www2018_results/dataset_twitter_agreement_" \
              "experiment_fixed/" \
              "crowd_train_merged_{}_labels.json".format(first_n)
        store_json(merged, dst)
        df = read_as_df(dst, tids)
        store_glove_vectors_for_dataset(GLOVE_VEC_TWITTER, df,
                                        glove_src)

    merged = merge_dicts(tweets, watson, rosette)
    set(merged.keys()) - set(tids)
    assert(len(set(merged.keys()) - set(tids)) == 0)
    df = dict_to_df(merged)

    # Preprocess tweets
    df = clean_tweets(df)
    file_name = os.path.basename(dst).split(".")[0]
    tfidf_path = os.path.join(tfidf_dir, "{}.json".format(file_name))
    # Transform data into matrix representation adding all features
    df = add_features(df, tfidf_path, is_training_set=True,
                      glove_src=glove_src, use_tfidf=use_tfidf)
    # Store as csv and json
    dst = os.path.join(csv_dir, "{}.csv".format(file_name))
    to_csv(df, features, LABEL_CLA, dst)
    dic = df.to_dict("index")
    dst = os.path.join(os.path.dirname(csv_dir), "{}.json".format(file_name))
    store_json(dic, dst)
    # read_exported_csv(dst)


def export_to_csv_combined(
        train, low, medium, high, dst, csv_dir, tfidf_dir, rosette_dir,
        watson_dir,
        rosette_dir_train, watson_dir_train, glove_src, use_tfidf, features,
        first_n, first_n_low):
    """
    Creates combined dataset of TRAINING, LOW, MEDIUM, HIGH.

    Parameters
    ----------
    train: str - path to the Twitter training set.
    low: str - path to LOW crowdsourced dataset.
    medium: str - path to MEDIUM crowdsourced dataset.
    high: str - path to HIGH crowdsourced dataset.
    dst: str - path where dataset will be stored.
    csv_dir: str - path to directory in which csv files will be stored.
    tfidf_dir: str - path to directory in which the tfidf_weights will be stored
    rosette_dir: str - path where the features extracted via Rosette are stored.
    watson_dir: str - path where the features extracted via Watson are stored.
    rosette_dir_train: str - path where the features extracted via Rosette are
    stored for the training set.
    watson_dir_train: str - path where the features extracted via Watson are
    stored for the training set
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's an empty string, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly.
    features: list of str - names of the features to be used for building the
    classifier.
    first_n: int - number of first labels to use from crowdsourcing datasets.
    first_n_low: int - number of first labels to use from crowdsourcing
    datasets for dataset LOW (because up to 8 labels are available).

    """
    low_tweets = read_amt_csv(low, first_n_low, 8)
    med_tweets = read_amt_csv(medium, first_n, 4)
    hig_tweets = read_amt_csv(high, first_n, 4)
    # Get tweet IDs of all tweets that were used in any of the 3 crowdsourcing
    # datasets, i.e. 3000, and the training set, in total 3.5k
    tweets = copy.deepcopy(low_tweets)
    tweets.update(med_tweets)
    tweets.update(hig_tweets)

    # Load training dataset
    with codecs.open(train, "rb", encoding="utf-8") as f:
        train_tweets = json.load(f, encoding="utf-8")
    # UPDATE labels of training set according to the definition:
    # if majority vote is <= 50% of the tweets, it's low agreement else high
    train_labels = get_agreement_and_sentiment_labels_training(train_tweets)
    for tid in train_tweets:
        train_tweets[tid][LABEL_CLA] = train_labels[tid][LABEL_CLA]
        train_tweets[tid][SENTIMENT_LABEL] = train_labels[tid][SENTIMENT_LABEL]
        train_tweets[tid][RELEVANCE_LABEL] = train_labels[tid][RELEVANCE_LABEL]
    tweets.update(train_tweets)
    for tid in tweets:
        assert(tweets[tid][LABEL_CLA] != "medium")
    tids = tweets.keys()

    # ##################################
    # # Use this to commented code to create datasets for Gizem
    # file_name = os.path.basename(dst).split(".")[0]
    # dst_gizem = os.path.join(csv_dir, "{}.csv".format(file_name))
    # with open(dst_gizem, "wb") as f:
    #     writer = csv.writer(f, delimiter="\t")
    #     for tid in tweets:
    #         writer.writerow([tid, tweets[tid][TEXT]])
    # #################################

    print "#total tweets", len(tweets)
    print "#tids", len(tids)
    # Merge Watson/Rosette features of training and crowdsourcing set
    watson = get_watson_data(watson_dir, tids)
    watson_train = get_watson_data(watson_dir_train, tids)
    watson.update(watson_train)
    rosette = get_rosette_data(rosette_dir, tids)
    rosette_train = get_rosette_data(rosette_dir_train, tids)
    rosette.update(rosette_train)

    # Run this function to extract only the glove vectors that exist in
    # our dataset -> then reading in GBs of text data isn't necessary and it's
    # much faster
    if not os.path.isfile(glove_src):
        print "watson", len(watson)
        print "rosette", len(rosette)
        merged = merge_dicts(tweets, watson, rosette)
        dst = "/media/data/Workspaces/PythonWorkspace/phd/Analyze-Labeled-" \
              "Dataset/www2018_results/dataset_twitter_agreement_" \
              "experiment_fixed/" \
              "crowd_train_merged_{}_labels.json".format(first_n)
        store_json(merged, dst)
        df = read_as_df(dst, tids)
        store_glove_vectors_for_dataset(GLOVE_VEC_TWITTER, df,
                                        glove_src)

    merged = merge_dicts(tweets, watson, rosette)
    set(merged.keys()) - set(tids)
    assert(len(set(merged.keys()) - set(tids)) == 0)
    df = dict_to_df(merged)

    # Preprocess tweets
    df = clean_tweets(df)
    file_name = os.path.basename(dst).split(".")[0]
    tfidf_path = os.path.join(tfidf_dir, "{}.json".format(file_name))
    # Transform data into matrix representation adding all features
    df = add_features(df, tfidf_path, is_training_set=True,
                      glove_src=glove_src, use_tfidf=use_tfidf)
    # Store as csv and json
    dst = os.path.join(csv_dir, "{}.csv".format(file_name))
    to_csv(df, features, LABEL_CLA, dst)
    dic = df.to_dict("index")
    dst = os.path.join(os.path.dirname(csv_dir), "{}.json".format(file_name))
    store_json(dic, dst)
    # read_exported_csv(dst)


def export_to_csv_low(
        low, dst, csv_dir, tfidf_dir, rosette_dir,
        watson_dir,
        rosette_dir_train, watson_dir_train, glove_src, use_tfidf, features,
        first_n):
    """
    Creates dataset for LOW.

    Parameters
    ----------
    low: str - path to LOW crowdsourced dataset.
    dst: str - path where dataset will be stored.
    csv_dir: str - path to directory in which csv files will be stored.
    tfidf_dir: str - path to directory in which the tfidf_weights will be stored
    rosette_dir: str - path where the features extracted via Rosette are stored.
    watson_dir: str - path where the features extracted via Watson are stored.
    rosette_dir_train: str - path where the features extracted via Rosette are
    stored for the training set.
    watson_dir_train: str - path where the features extracted via Watson are
    stored for the training set
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's an empty string, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly.
    features: list of str - names of the features to be used for building the
    classifier.
    first_n: int - number of first labels to use from crowdsourcing datasets.

    """
    low_tweets = read_amt_csv(low, first_n, 8)
    # Get tweet IDs of all tweets that were used in any of the 3 crowdsourcing
    # datasets, i.e. 3000, and the training set, in total 3.5k
    tweets = copy.deepcopy(low_tweets)
    # ##################################
    # # Use this to commented code to create datasets for Gizem
    # file_name = os.path.basename(dst).split(".")[0]
    # dst_gizem = os.path.join(csv_dir, "{}.csv".format(file_name))
    # with open(dst_gizem, "wb") as f:
    #     writer = csv.writer(f, delimiter="\t")
    #     for tid in tweets:
    #         writer.writerow([tid, tweets[tid][TEXT]])
    # #################################
    tids = tweets.keys()
    for tid in tweets:
        assert(tweets[tid][LABEL_CLA] != "medium")

    print "#total tweets", len(tweets)
    print "#tids", len(tids)
    # Merge Watson/Rosette features of training and crowdsourcing set
    watson = get_watson_data(watson_dir, tids)
    watson_train = get_watson_data(watson_dir_train, tids)
    watson.update(watson_train)
    rosette = get_rosette_data(rosette_dir, tids)
    rosette_train = get_rosette_data(rosette_dir_train, tids)
    rosette.update(rosette_train)

    # Run this function to extract only the glove vectors that exist in
    # our dataset -> then reading in GBs of text data isn't necessary and it's
    # much faster
    if not os.path.isfile(glove_src):
        # Get the absolute path to the parent directory of /scripts/
        base_dir = os.path.abspath(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), os.pardir))
        print "watson", len(watson)
        print "rosette", len(rosette)
        merged = merge_dicts(tweets, watson, rosette)
        dst = os.path.join(base_dir, "results",
                           "dataset_twitter_agreement_experiment",
                           "crowd_train_merged_{}_labels.json".format(first_n))
        store_json(merged, dst)
        df = read_as_df(dst, tids)
        store_glove_vectors_for_dataset(GLOVE_VEC_TWITTER, df,
                                        glove_src)

    merged = merge_dicts(tweets, watson, rosette)
    set(merged.keys()) - set(tids)
    assert(len(set(merged.keys()) - set(tids)) == 0)
    df = dict_to_df(merged)

    # Preprocess tweets
    df = clean_tweets(df)
    file_name = os.path.basename(dst).split(".")[0]
    tfidf_path = os.path.join(tfidf_dir, "{}.json".format(file_name))
    # Transform data into matrix representation adding all features
    df = add_features(df, tfidf_path, is_training_set=True,
                      glove_src=glove_src, use_tfidf=use_tfidf)
    # Store as csv and json
    dst = os.path.join(csv_dir, "{}.csv".format(file_name))
    to_csv(df, features, LABEL_CLA, dst)
    dic = df.to_dict("index")
    dst = os.path.join(os.path.dirname(csv_dir), "{}.json".format(file_name))
    store_json(dic, dst)
    # read_exported_csv(dst)


def read_exported_csv(src):
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            print row


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


def export_to_arff(src, gizem_path, header_path, dst_arff, features):
    """
    Exports combined dataset of TRAINING, LOW, MEDIUM, HIGH to arff file.

    Parameters
    ----------
    src: str - path to the json dataset that contains my extracted features.
    gizem_path: str - path to file in which Gizem's extracted features are.
    header_path: str - path to file that contains the names of the features in
    <gizem_path> in the same order.
    dst_arff: str - path where dataset as arff file with all extracted will be
    stored.
    features: list of str - names of the features to be used for building the
    classifier.

    """
    features_ = read_gizem_features(gizem_path, header_path)
    tweets = read_json(src)
    print "#total tweets", len(tweets)

    df = merge_to_df(tweets, features_)
    print "#features", df.shape
    to_arff(df, features, LABEL_CLA, dst_arff)


def export_train_to_csv(train, dst, csv_dir, tfidf_dir, rosette_dir_train,
                        watson_dir_train, glove_src, use_tfidf, features):
    """
    Creates dataset for TRAIN. This is only necessary because we want to compare
    binary agreement instead of 3 classes. Otherwise comparing TRAIN with LOW,
    MEDIUM, HIGH would be biased/unfair.


    Parameters
    ----------
    train: str - path to TRAIN dataset.
    dst: str - path where dataset will be stored.
    csv_dir: str - path to directory in which csv files will be stored.
    tfidf_dir: str - path to directory in which the tfidf_weights will be stored
    rosette_dir_train: str - path where the features extracted via Rosette are
    stored for the training set.
    watson_dir_train: str - path where the features extracted via Watson are
    stored for the training set
    glove_src: str - path to file storing glove vectors in json format for our
    dataset. If it's an empty string, no Glove vectors are used.
    use_tfidf: bool: True if glove vectors should be weighted according to their
    TF-IDF scores. Else they are weighted uniformly.
    features: list of str - names of the features to be used for building the
    classifier.

    """
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    tweets = read_json(train)
    # ##################################
    # # Use this to commented code to create datasets for Gizem
    # file_name = os.path.basename(dst).split(".")[0]
    # dst_gizem = os.path.join(csv_dir, "{}.csv".format(file_name))
    # with open(dst_gizem, "wb") as f:
    #     writer = csv.writer(f, delimiter="\t")
    #     for tid in tweets:
    #         writer.writerow([tid, tweets[tid][TEXT]])
    # #################################
    # UPDATE labels of training set according to the definition:
    # if majority vote is <= 50% of the tweets, it's low agreement else high
    train_labels = get_agreement_and_sentiment_labels_training(tweets)
    for tid in tweets:
        tweets[tid][LABEL_CLA] = train_labels[tid][LABEL_CLA]
        tweets[tid][SENTIMENT_LABEL] = train_labels[tid][SENTIMENT_LABEL]
        tweets[tid][RELEVANCE_LABEL] = train_labels[tid][RELEVANCE_LABEL]
    for tid in tweets:
        assert(tweets[tid][LABEL_CLA] != "medium")
    tids = tweets.keys()
    print "#total tweets", len(tweets)
    print "#tids", len(tids)
    watson = get_watson_data(watson_dir_train, tids)
    rosette = get_rosette_data(rosette_dir_train, tids)

    # Run this function to extract only the glove vectors that exist in
    # our dataset -> then reading in GBs of text data isn't necessary and it's
    # much faster
    if not os.path.isfile(glove_src):
        print "watson", len(watson)
        print "rosette", len(rosette)
        merged = merge_dicts(tweets, watson, rosette)
        dst = os.path.join(base_dir, "results",
                           "dataset_twitter_agreement_experiment",
                           "train_{}_labels.json".format("whatever"))
        store_json(merged, dst)
        df = read_as_df(dst, tids)
        store_glove_vectors_for_dataset(GLOVE_VEC_TWITTER, df,
                                        glove_src)

    merged = merge_dicts(tweets, watson, rosette)
    set(merged.keys()) - set(tids)
    assert(len(set(merged.keys()) - set(tids)) == 0)
    df = dict_to_df(merged)

    # Preprocess tweets
    df = clean_tweets(df)
    file_name = os.path.basename(dst).split(".")[0]
    tfidf_path = os.path.join(tfidf_dir, "{}.json".format(file_name))
    # Transform data into matrix representation adding all features
    df = add_features(df, tfidf_path, is_training_set=True,
                      glove_src=glove_src, use_tfidf=use_tfidf)
    # Store as csv and json
    dst = os.path.join(csv_dir, "{}.csv".format(file_name))
    to_csv(df, features, LABEL_CLA, dst)
    dic = df.to_dict("index")
    dst = os.path.join(os.path.dirname(csv_dir), "{}.json".format(file_name))
    store_json(dic, dst)
    # read_exported_csv(dst)


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Crowdsourced dataset containing only predicted low agreement tweets
    LOW = os.path.join(base_dir, "results", "dataset_twitter_crowdsourcing",
                       "Batch_2984090_batch_results_low_8000.csv")
    # Crowdsourced dataset containing only predicted medium agreement tweets
    MEDIUM = os.path.join(base_dir, "results", "dataset_twitter_crowdsourcing",
                          "Batch_2984078_batch_results_medium_4000.csv")
    # Crowdsourced dataset containing only predicted high agreement tweets
    HIGH = os.path.join(base_dir, "results", "dataset_twitter_crowdsourcing",
                        "Batch_2984071_batch_results_high_4000.csv")
    # Training set used to train the agreement classifier
    TRAIN = os.path.join(base_dir, "results", "dataset_twitter",
                         "dataset_twitter_watson_rosette_cleaned.json")
    # Directory in which the glove vectors existing in the twitter dataset
    # will be stored
    TWITTER_DIR = os.path.join(base_dir, "results", "glove_twitter")
    # Path to pretrained glove vectors
    # Downloaded from here:
    # https://github.com/stanfordnlp/GloVe
    # 200 dimensional vectors, uncased, 27B tokens, 2B tweets,
    # 1.2 Mio vocabulary from Twitter
    GLOVE_VEC_TWITTER = os.path.join(base_dir, "results", "glove",
                                     "glove.twitter.27B",
                                     "glove.twitter.27B.200d.txt")
    # Will only exist after running store_glove_vectors_for_dataset() once
    # Path to glove vectors existing in our dataset in w2v format.
    # Trained on Twitter
    GLOVE_W2V_VEC_TWITTER = os.path.join(TWITTER_DIR, "glove.twitter.27B.200d_glove_for_twitter_train_crowd_watson_rosette_min_annos_3_cleaned.json")
    # Glove vector using only training data
    GLOVE_W2V_VEC_TWITTER_TRAIN = os.path.join(TWITTER_DIR,
                                         "glove.twitter.27B.200d_glove_for_twitter_watson_rosette_min_annos_3_cleaned.json")
    # Directory in which datasets and data will be stored
    DST_DIR = os.path.join(base_dir, "results",
                           "dataset_twitter_agreement_experiment_fixed")
    # Directory in which the .csv files will be stored
    CSV_DIR = os.path.join(DST_DIR, "csv_files")
    # Directory in which computed TF-IDF vectors will be stored for the
    # different datasets
    TFIDF_DIR = os.path.join(DST_DIR, "tf_idf")
    ROSETTE_DIR = os.path.join(base_dir, "results",
                               "rosette_sentiment_twitter_full")
    WATSON_DIR = os.path.join(base_dir, "results",
                              "watson_sentiment_twitter_full")

    ROSETTE_DIR_TRAIN = os.path.join(base_dir, "results",
                                     "rosette_sentiment_twitter")
    WATSON_DIR_TRAIN = os.path.join(base_dir, "results",
                                    "watson_sentiment_twitter")
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
    if not os.path.exists(TWITTER_DIR):
        os.makedirs(TWITTER_DIR)
    if not os.path.exists(TFIDF_DIR):
        os.makedirs(TFIDF_DIR)
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
    if not os.path.exists(TFIDF_DIR):
        os.makedirs(TFIDF_DIR)

    # Use "cleaned" version of dataset, i.e. discard all other annotation times
    # if a tweet was assigned the label "Irrelevant"
    cleaned = True
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    use_tfidf = True

    ##################################################################
    # The following code is performed BEFORE having Gizem's features #
    ##################################################################

    ############################################################################
    ######################################################################
    # 1. Create merged dataset using 4 (MEDIUM, HIGH) and 4 labels (LOW) #
    ######################################################################
    # first_n = 4
    # first_n_low = 4
    # fname = "twitter_agreement_{}_labels_{}_low_labels.json".format(first_n,
    #                                                                 first_n_low)
    # dst = os.path.join(DST_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_csv_combined(TRAIN, LOW, MEDIUM, HIGH, dst, CSV_DIR, TFIDF_DIR,
    #                        ROSETTE_DIR, WATSON_DIR, ROSETTE_DIR_TRAIN,
    #                        WATSON_DIR_TRAIN, GLOVE_W2V_VEC_TWITTER, use_tfidf,
    #                        FEATURES, first_n, first_n_low)

    ######################################################################
    # 2. Create merged dataset using 4 (MEDIUM, HIGH) and 8 labels (LOW) #
    ######################################################################
    # first_n = 4
    # first_n_low = 8
    # fname = "twitter_agreement_{}_labels_{}_low_labels.json".format(first_n,
    #                                                                 first_n_low)
    # dst = os.path.join(DST_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_csv_combined(TRAIN, LOW, MEDIUM, HIGH, dst, CSV_DIR, TFIDF_DIR,
    #                        ROSETTE_DIR, WATSON_DIR, ROSETTE_DIR_TRAIN,
    #                        WATSON_DIR_TRAIN, GLOVE_W2V_VEC_TWITTER, use_tfidf,
    #                        FEATURES, first_n, first_n_low)

    ########################################
    # 3. Create LOW dataset using 4 labels #
    ########################################
    # first_n = 4
    # fname = "twitter_agreement_low_{}_labels.json".format(first_n)
    # dst = os.path.join(DST_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_csv_low(LOW, dst, CSV_DIR, TFIDF_DIR, ROSETTE_DIR, WATSON_DIR,
    #                   ROSETTE_DIR_TRAIN, WATSON_DIR_TRAIN,
    #                   GLOVE_W2V_VEC_TWITTER, use_tfidf, FEATURES, first_n)

    ########################################
    # 4. Create LOW dataset using 8 labels #
    ########################################
    # first_n = 8
    # fname = "twitter_agreement_low_{}_labels.json".format(first_n)
    # dst = os.path.join(DST_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_csv_low(LOW, dst, CSV_DIR, TFIDF_DIR, ROSETTE_DIR, WATSON_DIR,
    #                   ROSETTE_DIR_TRAIN, WATSON_DIR_TRAIN,
    #                   GLOVE_W2V_VEC_TWITTER, use_tfidf, FEATURES, first_n)

    ##################################################################
    # 5. Create TRAIN dataset with updated (binary) agreement labels #
    ##################################################################
    fname = "twitter_agreement_train.json"
    dst = os.path.join(DST_DIR, fname)
    # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # glove vectors
    export_train_to_csv(TRAIN, dst, CSV_DIR, TFIDF_DIR,
                        ROSETTE_DIR_TRAIN, WATSON_DIR_TRAIN,
                        GLOVE_W2V_VEC_TWITTER_TRAIN, use_tfidf, FEATURES)

    #####################################################
    # 6. Merge LOW, MEDIUM, HIGH using 4 labels for LOW #
    #####################################################
    # first_n = 4
    # first_n_low = 4
    # fname = "twitter_agreement_crowd_{}_labels_{}_low_labels.json"\
    #     .format(first_n, first_n_low)
    # dst = os.path.join(DST_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_csv_crowd(LOW, MEDIUM, HIGH, dst, CSV_DIR, TFIDF_DIR, ROSETTE_DIR,
    #                     WATSON_DIR, ROSETTE_DIR_TRAIN, WATSON_DIR_TRAIN,
    #                     GLOVE_W2V_VEC_TWITTER, use_tfidf, FEATURES, first_n,
    #                     first_n_low)

    #####################################################
    # 7. Merge LOW, MEDIUM, HIGH using 8 labels for LOW #
    #####################################################
    # first_n = 4
    # first_n_low = 8
    # fname = "twitter_agreement_crowd_{}_labels_{}_low_labels.json" \
    #     .format(first_n, first_n_low)
    # dst = os.path.join(DST_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_csv_crowd(LOW, MEDIUM, HIGH, dst, CSV_DIR, TFIDF_DIR, ROSETTE_DIR,
    #                     WATSON_DIR, ROSETTE_DIR_TRAIN, WATSON_DIR_TRAIN,
    #                     GLOVE_W2V_VEC_TWITTER, use_tfidf, FEATURES, first_n,
    #                     first_n_low)
    ############################################################################

    #####################################################
    # The next part is run when having Gizem's features #
    #####################################################

    ############################################################################
    global FEATURES
    FEATURES.extend(GIZEMS_NAMES)
    ######################################################################
    # 1. Create merged dataset using 4 (MEDIUM, HIGH) and 4 labels (LOW) #
    ######################################################################
    # # Use Gizem's features too
    # # Path to combined dataset with my extracted features (that was created in
    # # the 1st part of the script
    # SRC = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "twitter_agreement_4_labels_4_low_labels.json")
    # # Path to Gizem's extracted features
    # FEATURE = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "FeatureFile_Trump_twitter_agreement_4_labels_4_low_labels.txt")
    # # Path to header file containing the feature names and their order
    # HEADER = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "Features_Trump.txt")
    # ARFF_DIR = os.path.join(base_dir, "results", "arff_files")
    # first_n = 4
    # first_n_low = 4
    # fname = "twitter_agreement_complete_{}_labels_{}_low_labels_fixed.arff"\
    #     .format(first_n, first_n_low)
    # dst = os.path.join(ARFF_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_arff(SRC, FEATURE, HEADER, dst, FEATURES)

    ######################################################################
    # 2. Create merged dataset using 4 (MEDIUM, HIGH) and 8 labels (LOW) #
    ######################################################################
    # Use Gizem's features too
    # Path to combined dataset with my extracted features (that wascreated in
    # the 1st part of the script
    # SRC = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "twitter_agreement_4_labels_8_low_labels.json")
    # # Path to Gizem's extracted features
    # FEATURE = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "FeatureFile_Trump_twitter_agreement_4_labels_8_low_labels.txt")
    # # Path to header file containing the feature names and their order
    # HEADER = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "Features_Trump.txt")
    # ARFF_DIR = os.path.join(base_dir, "results", "arff_files")
    # first_n = 4
    # first_n_low = 8
    # fname = "twitter_agreement_complete_{}_labels_{}_low_labels_fixed.arff"\
    #     .format(first_n, first_n_low)
    # dst = os.path.join(ARFF_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_arff(SRC, FEATURE, HEADER, dst, FEATURES)

    ########################################
    # 3. Create LOW dataset using 4 labels #
    ########################################
    # Path to combined dataset with my extracted features (that wascreated in
    # the 1st part of the script
    # SRC = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "twitter_agreement_low_4_labels.json")
    # # Path to Gizem's extracted features
    # FEATURE = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "FeatureFile_Trump_twitter_agreement_low_4_labels.txt")
    # # Path to header file containing the feature names and their order
    # HEADER = "results", "dataset_twitter_agreement_experiment_fixed", "Features_Trump.txt")
    # ARFF_DIR = os.path.join(base_dir, "results", "arff_files")
    # first_n = 4
    # fname = "twitter_agreement_low_complete_{}_labels_fixed.arff"\
    #     .format(first_n)
    # dst = os.path.join(ARFF_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_arff(SRC, FEATURE, HEADER, dst, FEATURES)

    ########################################
    # 4. Create LOW dataset using 8 labels #
    ########################################
    # Path to combined dataset with my extracted features (that wascreated in
    # the 1st part of the script
    # SRC = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "twitter_agreement_low_8_labels.json")
    # # Path to Gizem's extracted features
    # FEATURE = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "FeatureFile_Trump_twitter_agreement_low_8_labels.txt")
    # # Path to header file containing the feature names and their order
    # HEADER = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "Features_Trump.txt")
    # ARFF_DIR = os.path.join(base_dir, "results", "arff_files")
    # first_n = 8
    # fname = "twitter_agreement_low_complete_{}_labels_fixed.arff"\
    #     .format(first_n)
    # dst = os.path.join(ARFF_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_arff(SRC, FEATURE, HEADER, dst, FEATURES)

    ##################################################################
    # 5. Create TRAIN dataset with updated (binary) agreement labels #
    ##################################################################
    # Use Gizem's features too
    # Path to combined dataset with my extracted features (that wascreated in
    # the 1st part of the script
    # SRC = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "twitter_agreement_train.json")
    # # Path to Gizem's extracted features
    # FEATURE = os.path.join(base_dir, "results", "dataset_twitter", "FeatureFile_Trump_Labelled_500.txt")
    # # Path to header file containing the feature names and their order
    # HEADER = os.path.join(base_dir, "results", "dataset_twitter", "Features_Trump.txt")
    # ARFF_DIR = "results/arff_files/")
    # first_n = 4
    # first_n_low = 8
    # fname = "twitter_agreement_train_fixed.arff"
    # dst = os.path.join(ARFF_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_arff(SRC, FEATURE, HEADER, dst, FEATURES)

    ########################################################################
    # 6. Create merged dataset of LOW, MEDIUM, HIGH using 4 labels for LOW #
    ########################################################################
    # SRC = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "twitter_agreement_crowd_4_labels_4_low_labels.json")
    # # Path to Gizem's extracted features
    # FEATURE = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "FeatureFile_Trump_twitter_agreement_crowd_4_labels_4_low_labels.txt")
    # # Path to header file containing the feature names and their order
    # HEADER = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "Features_Trump.txt")
    # ARFF_DIR = os.path.join(base_dir, "results", "arff_files")
    # first_n = 4
    # first_n_low = 4
    # fname = "twitter_agreement_crowd_{}_labels_{}_low_labels_fixed.arff" \
    #     .format(first_n, first_n_low)
    # dst = os.path.join(ARFF_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_arff(SRC, FEATURE, HEADER, dst, FEATURES)

    ########################################################################
    # 7. Create merged dataset of LOW, MEDIUM, HIGH using 8 labels for LOW #
    ########################################################################
    # SRC = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "twitter_agreement_crowd_4_labels_8_low_labels.json")
    # # Path to Gizem's extracted features
    # FEATURE = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "FeatureFile_Trump_twitter_agreement_crowd_4_labels_8_low_labels.txt")
    # # Path to header file containing the feature names and their order
    # HEADER = os.path.join(base_dir, "results", "dataset_twitter_agreement_experiment_fixed", "Features_Trump.txt")
    # ARFF_DIR = os.path.join(base_dir, "results", "arff_files")
    # first_n = 4
    # first_n_low = 8
    # fname = "twitter_agreement_crowd_{}_labels_{}_low_labels_fixed.arff"\
    #     .format(first_n, first_n_low)
    # dst = os.path.join(ARFF_DIR, fname)
    # # Create combined dataset (TRAINING + LOW + MEDIUM + HIGH) using Twitter
    # # glove vectors
    # export_to_arff(SRC, FEATURE, HEADER, dst, FEATURES)
    ###########################################################################
