"""
Use IBM Watson to extract sentiment and NERs from tweets
"""
#  https://www.ibm.com/watson/developercloud/natural-language-understanding/api/v1/#sentiment
import json
import codecs
import os
import csv
import random

import watson_developer_cloud
from watson_developer_cloud import NaturalLanguageUnderstandingV1
import watson_developer_cloud.natural_language_understanding.features.v1 \
    as Features

from create_dataset_twitter import read_dataset, aggregate_data_per_tweet

# Credentials
natural_language_understanding = NaturalLanguageUnderstandingV1(
  username="43b6f2b7-035f-4c42-907a-dbde461bd699",
  password="UeabUkIFYPCk",
  version="2017-02-27")


# Labels that were assigned by experts and also exist in crowdsourced dataset
# (plus the label "I can't judge")
LABEL_MAPPING = {
    "0": "Not Relevant",
    "1": "Relevant",
    "2": "Highly Relevant"
}

SEED = 13
random.seed(SEED)


def read_twitter(cleaned, min_annos):
    """
    Read our Twitter dataset.

    Parameters
    ----------
    cleaned: bool - True if only cleaned data should be used (i.e. any
    additional labels (and their annotation times) assigned to tweets considered
    "Irrelevant" are ignored)
    min_annos: int -  minimum number of annotators who must've labeled a tweet
    for it to be considered.

    Returns
    --------
    dict.

    """
    # Names of the collections in the DB
    anno_coll_name = "user"
    tweet_coll_name = "tweets"

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
    data = aggregate_data_per_tweet(annos, counts)
    # Remove all line breaks in a tweet text
    for tid in data:
        tweet = data[tid]

        tweet["text"] = tweet["text"].replace('\n', ' ').replace('\r', '')
    return data


def read_trec(src):
    """
    Read TREC dataset.

    Parameters
    ----------
    src: str - path to input file.

    Returns
    --------
    dict.
    # {tid: text}

    """
    data = {}
    with open(src, "rb") as csvfile:
        csv_reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        # Skip header
        for i, row in enumerate(csv_reader):
            # Skip header
            if i > 0:
                # Extract data
                # print row
                tid = row[0]
                text = row[-2]
                if (i-1) % 3 == 0:
                    data[tid] = text
    return data


def extract_sentiment_ner_twitter(cleaned, min_annos, dst_dir):
    """
    Extracts tweet overall sentiment, sentiment per NER, NERs,
    keywords, sentiment per keyword.

    Parameters
    ----------
    cleaned: bool - True if only cleaned data should be used (i.e. any
    additional labels (and their annotation times) assigned to tweets considered
    "Irrelevant" are ignored)
    min_annos: int -  minimum number of annotators who must've labeled a tweet
    for it to be considered.
    dst_dir: - directory in which results will be stored.

    """
    tweets = read_twitter(cleaned, min_annos)
    for idx, tid in enumerate(tweets):
        tweet = tweets[tid]
        fname = "{}.json".format(tid)
        dst = os.path.join(dst_dir, fname)
        print tweet["text"]
        # If file already exists, data was extracted before and due to
        # rate-limiting the rest couldn't be extracted
        if not os.path.isfile(dst):
            # Extract features for a tweet via Watson
            response = natural_language_understanding.analyze(
                text=tweet["text"],
                # Get entities and their
                features=[
                    # Overall tweet sentiment
                    Features.Sentiment(),
                    # NER detection and sentiment per NER
                    Features.Entities(sentiment=True),
                    Features.Keywords(sentiment=True),
                ]
            )

            # Store results in UTF-8 encoding
            fname = "{}.json".format(tid)
            dst = os.path.join(dst_dir, fname)
            with codecs.open(dst, "w", encoding="utf-8") as f:
                # https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
                data = json.dumps(response, ensure_ascii=False, encoding='utf8')
                f.write(unicode(data))
        print "Finished extraction for {} tweets".format(idx+1)


def extract_sentiment_ner_trec(src, dst_dir):
    """
    Extracts tweet overall sentiment, sentiment per NER, NERs,
    keywords, sentiment per keyword.

    Parameters
    ----------
    src: str - path to dataset.
    dst_dir: - directory in which results will be stored.

    """
    tweets = read_trec(src)
    for idx, tid in enumerate(tweets):
        fname = "{}.json".format(tid)
        dst = os.path.join(dst_dir, fname)
        # If file already exists, data was extracted before and due to
        # rate-limiting the rest couldn't be extracted
        if not os.path.isfile(dst):
            try:
                # Extract features for a tweet via Watson
                response = natural_language_understanding.analyze(
                    text=tweets[tid],
                    # Get entities and their
                    features=[
                        # Overall tweet sentiment
                        Features.Sentiment(),
                        # NER detection and sentiment per NER
                        Features.Entities(sentiment=True),
                        Features.Keywords(sentiment=True),
                    ]
                )

                # Store results in UTF-8 encoding
                fname = "{}.json".format(tid)
                dst = os.path.join(dst_dir, fname)
                with codecs.open(dst, "w", encoding="utf-8") as f:
                    # https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
                    data = json.dumps(response, ensure_ascii=False, encoding='utf8')
                    f.write(unicode(data))
            # Illegal language
            except watson_developer_cloud.watson_developer_cloud_service.\
                    WatsonException:
                pass
        print "Finished extraction for {} tweets".format(idx+1)


def read_txt(src):
    """
    Reads the expert labels and text from a .txt file.

    Parameters
    ----------
    src: str - path to input file.

    Returns
    -------
    dict.
    {tid:
        {
            "text": ...,
            "label": ...
        }
    }
    Tweet ID as key and text and expert label as values in inner dictionary.

    """
    # {tid:
    #     {
    #         "text": ...,
    #         "label": ...
    #     }
    # }
    tweets = {}
    with open(src, "rb") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        try:
            tid, user, text, topic_id, label_id = line.split("\t")
            # There was at least 1 tweet with expert label -2
            if label_id in LABEL_MAPPING:
                tweets[tid] = {}
                tweets[tid]["text"] = text
                tweets[tid]["label"] = LABEL_MAPPING[label_id]
            # There was at least 1 tweet (ID: 30252178127986688) that
            # contained "\t" in the text, so, it can't be parsed.
        except ValueError:
            pass
    return tweets


def extract_sentiment_ner_trec_full(src, dst_dir):
    """
    Extracts tweet overall sentiment, sentiment per NER, NERs,
    keywords, sentiment per keyword  for the full dataset that's read from a
    .txt file.

    Parameters
    ----------
    src: str - path to dataset.
    dst_dir: - directory in which results will be stored.

    """
    tweets = read_txt(src)
    # Since tweets are ordered according to topic, label them in a
    # random order
    keys = tweets.keys()
    random.shuffle(keys)

    for idx, tid in enumerate(keys):
        fname = "{}.json".format(tid)
        dst = os.path.join(dst_dir, fname)
        # If file already exists, data was extracted before and due to
        # rate-limiting the rest couldn't be extracted
        if not os.path.isfile(dst):
            try:
                # Extract features for a tweet via Watson
                response = natural_language_understanding.analyze(
                    text=tweets[tid]["text"],
                    # Get entities and their
                    features=[
                        # Overall tweet sentiment
                        Features.Sentiment(),
                        # NER detection and sentiment per NER
                        Features.Entities(sentiment=False),
                        Features.Keywords(sentiment=False),
                    ]
                )

                # Store results in UTF-8 encoding
                fname = "{}.json".format(tid)
                dst = os.path.join(dst_dir, fname)
                with codecs.open(dst, "w", encoding="utf-8") as f:
                    # https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
                    data = json.dumps(response, ensure_ascii=False,
                                      encoding='utf8')
                    f.write(unicode(data))
            # Illegal language
            except watson_developer_cloud.watson_developer_cloud_service.\
                    WatsonException:
                pass
        print "Finished extraction for {} tweets".format(idx+1)


def read_tweets_csv(src):
    """
    Reads tweets from csv file.

    Parameters
    ----------
    src: str - path to csv dataset.

    Returns
    -------
    dict.
    {tid: text}.

    """
    tweets = {}
    with open(src, "rb") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            tid, text = row
            tweets[tid] = text
    return tweets


def extract_sentiment_ner_twitter_full(src, dst_dir):
    """
    Extracts tweet overall sentiment, sentiment per NER, NERs,
    POS tags for the full dataset that's read from a .csv file.

    Parameters
    ----------
    src: str - path to dataset.
    dst_dir: - directory in which results will be stored.

    """
    tweets = read_tweets_csv(src)
    print "#tweets", len(tweets)
    for idx, tid in enumerate(tweets):
        fname = "{}.json".format(tid)
        dst = os.path.join(dst_dir, fname)
        # If file already exists, data was extracted before and due to
        # rate-limiting the rest couldn't be extracted
        if not os.path.isfile(dst):
            try:
                # Extract features for a tweet via Watson
                response = natural_language_understanding.analyze(
                    text=tweets[tid],
                    # Get entities and their
                    features=[
                        # Overall tweet sentiment
                        Features.Sentiment(),
                        # NER detection and sentiment per NER
                        Features.Entities(sentiment=True),
                        Features.Keywords(sentiment=True),
                    ]
                )

                # Store results in UTF-8 encoding
                fname = "{}.json".format(tid)
                dst = os.path.join(dst_dir, fname)
                with codecs.open(dst, "w", encoding="utf-8") as f:
                    # https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
                    data = json.dumps(response, ensure_ascii=False, encoding='utf8')
                    f.write(unicode(data))
            except watson_developer_cloud.watson_developer_cloud_service.WatsonException:
                pass
        print "Finished extraction for {} tweets".format(idx+1)


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    ##################
    # 1. Once for our collected dataset
    ##################
    # Directory in which results will be stored.
    DST_DIR = os.path.join(base_dir, "results", "watson_sentiment_twitter")
    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
    cleaned = True
    min_annos = 3
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    extract_sentiment_ner_twitter(cleaned, min_annos, DST_DIR)

    ##################
    # 4. Once for the full Twitter dataset
    ##################
    # Directory in which results will be stored
    DST_DIR = os.path.join(base_dir, "results", "watson_sentiment_twitter_full")
    # Directory in which the dataset is stored
    DS_DIR = os.path.join(base_dir, "results", "dataset_twitter_full")
    if not os.path.exists(DST_DIR):
        os.makedirs(DST_DIR)
    cleaned = True
    min_annos = 3
    agg = "raw"
    if cleaned:
        agg = "cleaned"
    fname = "tweets.csv"
    src = os.path.join(DS_DIR, fname)
    if not os.path.isfile(src):
        raise IOError("Dataset doesn't exist!")
    extract_sentiment_ner_twitter_full(src, DST_DIR)
