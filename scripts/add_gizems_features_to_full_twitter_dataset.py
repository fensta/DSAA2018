"""
For some tweets of the full Twitter dataset there are no Watson and/or Rosette
features or some other features from Gizem might be missing. Thus, we only keep
valid ones.

"""
import os
import codecs
import json
import unicodecsv as csv


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


if __name__ == "__main__":
    # Get the absolute path to the parent directory of /scripts/
    base_dir = os.path.abspath(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), os.pardir))
    # Path where the merged dataset will be stored
    DST = os.path.join(base_dir, "results", "dataset_twitter_full",
                       "twitter_full_gizem_features.json")
    # Path to Gizem's extracted features
    FEATURES = os.path.join(base_dir, "results", "dataset_twitter_full",
                            "FeatureFile_Trump_UnLabelled_25451.txt")
    # Path to header file containing the feature names and their order
    HEADER = os.path.join(base_dir, "results", "dataset_twitter",
                          "Features_Trump.txt")
    # Path to Rosette features
    ROSETTE_DIR = os.path.join(base_dir, "results",
                               "rosette_sentiment_twitter_full")
    # Path to Watson features
    WATSON_DIR = os.path.join(base_dir, "results",
                              "watson_sentiment_twitter_full")
    # Directory in which the full Twitter dataset is stored
    FULL_DIR = os.path.join(base_dir, "results", "dataset_twitter_full",
                            "tweets.csv")

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
    data = read_twitter_csv(FULL_DIR)

    # Store updated dataset with expert labels and only consider tweets for
    # which we have Rosette and Watson features
    # {tid:
    #      {
    #          "text": "...",
    #      }
    # }
    tweets = {}
    with codecs.open(DST, "w", encoding="utf-8") as f:
        for tid in data:
            # Only consider tweets for which we have all features
            if tid in watson and tid in rosette and tid in features:
                text = data[tid]
                # Remove all line breaks in a tweet
                text = text.replace('\n', ' ').replace('\r', '')
                # text = text.decode("utf-8")
                tweets[tid] = {
                    "text": text,
                }
                # Add Gizem's features
                tweets[tid].update(features[tid])
        # https://stackoverflow.com/questions/18337407/saving-utf-8-texts-in-json-dumps-as-utf8-not-as-u-escape-sequence
        data = json.dumps(tweets, encoding="utf-8", indent=4,
                          ensure_ascii=False)
        f.writelines(unicode(data))
    data = read_gizem_json(DST)
    print "#tweets", len(data)
