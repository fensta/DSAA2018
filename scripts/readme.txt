########################################################################################
# 1. Using Twitter dataset to build three datasets for crowdsourcing -> Experiment 1+2 #
########################################################################################
1. add_gizems_features_to_twitter_dataset.py --> add Gizem's extracted features to training set and remove some tweets for which not all features are available
2. add_gizems_features_to_full_twitter_dataset.py --> add Gizem's extracted features to test set and remove some tweets for which not all features are available
3. create_twitter_test_set.py --> create csv file for all the unlabeled Trump tweets that weren't labeled
4. extract_sentiment_ner_watson.py --> extract tweet sentiment and NERs from tweets using Watson (these features are used in the classifier and must therefore be extracted prior to building it)
5. extract_sentiment_ner_rosette.py --> extract tweet sentiment and NERs from tweets using Rosette (these features are used in the classifier and must therefore be extracted prior to building it). Rosette is trained on tweets(https://www.quora.com/Is-there-any-tool-or-API-trained-on-tweet-data-for-named-entity-extraction), so it might be more accurate
6. create_dataset_twitter.py --> create files used for creating training and test sets for sentiment analysis in the next steps
7. build_classifier_twitter_sentiment_relevance.py --> arff file used for predicting if tweet is relevant. Predictions are stored under weka_predictions/sentiment_relevance_predictions_twitter
< use /arff_files/dataset_twitter_relevance_sentiment_twitter_classification.arff with 10-fold CV to obtain Weka predictions and store them as /weka_predictions/sentiment_relevance_predictions_twitter/sentiment_relevance_twitter_training.csv>
8. build_classifier_twitter_sentiment_sentiment.py --> arff file used for predicting the sentiment of all tweets which were considered relevant in the previous step. Predictions are stored under weka_predictions/sentiment_sentiment_predictions_twitter
< use /arff_files/dataset_twitter_sentiment_sentiment_twitter_classification.arff with 10-fold CV to obtain Weka predictions and store them as /weka_predictions/sentiment_sentiment_predictions_twitter/sentiment_sentiment_twitter_training.csv>>
9. build_classifier_twitter_final.py --> add 2 classifier-related features for agreement prediction to training set
< use /arff_files/dataset_twitter_agreement_final_twitter_classification.arff as training set and /arff_files/dataset_twitter_full_agreement_final_twitter_classification.arff as test set to obtain Weka predictions and store them as /weka_predictions/agreement_predictions_twitter/agreement_twitter_test.csv>>
10. build_crowdsourcing_datasets.py

######################################################################
# 2. Compute inter-annotator agreement within each of the 3 datasets #
######################################################################
1. compute_agreement_per_dataset.py --> analyze agreement by plots and compute some statistics; figures from our paper are in the directory results/figures/ --> Fig. 3 = "crowdsourcing_label_distribution_8_labels_multiple.pdf", Fig. 5 = "crowdsourcing_agreement_distribution_4_labels_multiple.pdf", Fig. 7 = "crowdsourcing_low_agreement_over_all_datasets.pdf"
2a. prepare_datasets_for_agreement_experiment_fixed.py --> extract features for 1 dataset comprising training set + 3 crowdsourced datasets, extract features for LOW dataset with 4 labels and LOW with 8 labels and send the datasets to Gizem for her to compute the features
2b. prepare_datasets_for_agreement_experiment_fixed.py --> run bottom part after obtaining gizem's features to merge them with my features and construct arff files

########################################################################################
# 3. Analyze impact of low agreement tweets on classifier performance --> Experiment 3 #
########################################################################################
1. export_majority_labels_gizem.py --> export majority labels from Twitter training set to csv, so that Gizem has access to the data
2a. extract_features_for_gizems_experiment.py --> extract for the datasets that Gizem generated the features and send the updated data to her
2b. extract_features_for_gizems_experiment.py --> run bottom part after obtaining gizem's features to merge them with my features and construct arff files
3. plot_gizems_experiment.py --> plot Gizem's results --> Fig. 6 = "influence_low_agreement_on_classifier_opti_2200.pdf"

Sentiment analysis comprises 2 steps: if tweet is relevant, predict its sentiment
---------------------------------------------------------------------------------
1. create_dataset_twitter.py
2. build_classifier_twitter_sentiment_relevance.py --> arff file used for predicting if tweet is relevant. Predictions are stored under weka_predictions/sentiment_relevance_predictions_twitter
3. build_classifier_twitter_sentiment_sentiment.py --> arff file used for predicting the sentiment of all tweets who were considered relevant in the previous step. Predictions are stored under weka_predictions/sentiment_sentiment_predictions_twitter
4. build_classifier_twitter_final.py --> add 2 classifier-related features for agreement prediction to training set

###########################################################################################
# 4. Extension of experiment 3 (effect of using more labels for tweets with low agreement #
###########################################################################################
1. prepare_datasets_for_additional_experiment.py --> counts how many tweets always (when using 4, 5, 6, 7, and 8 labels) exhibit high/low disagreement; also extracts tweets for Gizem (uncomment the small part before merge_tweets())
2. extract_features_for_additional_experiment.py --> run bottom part for merging Gizem's features with mine from which the arff files will be generated
3. plot_additional_experiment.py --> plot resulting AUC curves --> Fig. 8 = "influence_low_agreement_on_classifier_opti_174.pdf"

#########################################################################################
# 5. Manual analysis of tweets with high disagreement (are they difficult or ambiguous? #
#########################################################################################
1. create_ambiguous_difficult_dataset.py --> create the dataset containing tweets with high disagreement
2. plot_difficulty_label_distribution.py --> plot difficulty label distribution --> Fig. 4 = "difficulty_label_distribution_all_5_labels_full_names.pdf"

##################################################
# 6. Label all tweets w.r.t. difficulty manually #
##################################################
1. create_difficult_dataset.py --> create dataset containing all tweets from TRAIN, LOW, MEDIUM, HIGH


####################################################################################
# 7. Correlation difficulty definition (majority labels <= 50%) vs. labels from 6. #
####################################################################################
1. compute_difficulty_definition_correlation.py

#######################################################
# 8. Extract Weka predictions for worker disagreement #
#######################################################
1. analyze_difficulty_predictions.py