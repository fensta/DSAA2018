/* DSAA Conference Paper, 2018 - Predicting worker disagreement for more effective crowd labeling
 * 
 * Maven Project implemented in Java on Eclipse
 * Written by Gizem Gezici, PHD Candidate
 * Sabanci University, Istanbul, Turkey
 * Last Updated: 25.07.2018 (making comments, re-factoring the code etc.)
 * 
 */


import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import org.json.simple.parser.ParseException;

import com.google.gson.JsonElement;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;


public class Generate_Features {

	public static List<String> StopWords = new ArrayList<String>();

	public static List<String> PosWords = new ArrayList<String>();
	public static List<String> NegWords = new ArrayList<String>();
	public static List<String> LingoWords = new ArrayList<String>();
	public static LinkedHashMap<String, SentiToken> SentiWordNet = new LinkedHashMap<String, SentiToken>();
	public static LinkedHashMap<String, Double> TwitterLexicon = new LinkedHashMap<String, Double>();

	public static List<List<Double>> TopicProbabilities = new ArrayList<List<Double>>();
	public static LinkedHashMap<String, BlobInstance> TextBlobOverallSentimentScores = new LinkedHashMap<String, BlobInstance>();
	public static LinkedHashMap<String, BlobInstance> TextBlobOverallSentimentScoresFirst = new LinkedHashMap<String, BlobInstance>();
	public static LinkedHashMap<String, BlobInstance> TextBlobOverallSentimentScoresSecond = new LinkedHashMap<String, BlobInstance>();

	public static LinkedHashMap<String, Double> FrequencyScores = new LinkedHashMap<String, Double>();



	public static int sumExcMarks = 0;
	public static int sumQuestMarks = 0;
	public static int sumPosEmots = 0;
	public static int sumNegEmots = 0;
	public static int sumSuspPoints = 0;

	public static int sumAllUppercaseTokens = 0;
	public static int sumRepeatingCharsTokens = 0;

	public static double sumAvgPolSenti = 0.0;
	public static double sumDomPolSenti = 0.0;
	public static double sumAvgPol = 0.0;
	public static double sumTextBlobSentiment = 0.0;

	public static double sumAvgPolSentiFirst = 0.0;
	public static double sumDomPolSentiFirst = 0.0;
	public static double sumAvgPolFirst = 0.0;
	public static double sumTextBlobSentimentFirst = 0.0;

	public static double sumAvgPolSentiSecond = 0.0;
	public static double sumDomPolSentiSecond = 0.0;
	public static double sumAvgPolSecond = 0.0;
	public static double sumTextBlobSentimentSecond = 0.0;

	public static String query = "donald trump hillary clinton political election discussion";
	//Smallest window size in which all of the terms appear in the document (tweet).
	//This is used for query-term proximity feature computation.
	public static int wSize = 10;
	public static String [] qTokens = query.split("\\s+");

	public static double qsumFreq = 0.0;
	public static double qmeanFreq = 0.0;
	public static double qminFreq = 0.0;
	public static double qmaxFreq = 0.0;
	public static double qvarianceFreq = 0.0;


	public static List<Tweet> TweetLists = new ArrayList<Tweet>();
	public static List<Instance> Instances = new ArrayList<Instance>();

	public static void main(String[] args) throws FileNotFoundException 
	{
		RunDemo();
	}
	//We read the dataset files in this method.
	static void GetTextBlob_LDAFiles()
	{
		String LDAFile = "";
		String LDAFile1 = "";
		String LDAFile2 = "";
		
		String datasetFile = "";
		String filePart = "";
		File folder = new File("StefanFiles_HCOMP/dataset_splits/LA_Tweets_Percentage_2200/");
		//Get all the files in the folder to be processed (id-tab-tweet)
		File[] listOfFiles = folder.listFiles();			
		//For each file in the dataset to be processed (on each line there is 1 instance as "id-tab-tweet")
		for (int i = 0; i < listOfFiles.length; i++) 
		{
			File currFile = listOfFiles[i];
			if (currFile.isFile()) 
			{	
				datasetFile = listOfFiles[i].getName();
				//For other experiments, there are also files with .csv extension
				filePart = datasetFile.replace(".csv", "");
				filePart = filePart.replace(".txt", "");
				
				//Read the dataset file and put all the tweets in the list of "TweetLists".
				ReadDataset("StefanFiles_HCOMP/dataset_splits/LA_Tweets_Percentage_2200/" + datasetFile);
				
				//In the FIRST run: These files will be used for LDA to get topic distribution and TextBlob 
				//processing to obtain sentiment information for each sentence in the current dataset file.
				LDAFile = "StefanFiles_HCOMP/for_lda/LDADocs_2200/LDADocs_2200_1/LDA_Document_Trump_" + filePart + ".txt";
				LDAFile1 = "StefanFiles_HCOMP/for_lda/LDADocs_2200/LDADocs_2200_1/LDA_DocumentFirst_Trump_" + filePart + ".txt";
				LDAFile2 = "StefanFiles_HCOMP/for_lda/LDADocs_2200/LDADocs_2200_1/LDA_DocumentSecond_Trump_" + filePart + ".txt";
				

				try {
					PrintWriter pw = new PrintWriter(LDAFile);
					PrintWriter pw1 = new PrintWriter(LDAFile1);
					PrintWriter pw2 = new PrintWriter(LDAFile2);
					
					int length = 0;
					
					String ID = "";
					String tweet = "";
					String firstTweet = "";
					String secondTweet = "";
					
					String [] tTokensFirstHalf = null;
					String [] tTokensSecondHalf = null;
					for (Tweet t : TweetLists) 
					{	
						ID = t.ID;
						tweet = t.text;
						firstTweet = "";
						secondTweet = "";	
						
						
						tweet = tweet.replaceAll("http://[^\\s]+", "");
						
						tweet = RemoveStopWords(tweet);
						tweet = tweet.toLowerCase(Locale.ENGLISH);
						String[] tTokens = tweet.split("\\s+");
						
						tTokens = tweet.replaceAll("[^a-zA-Z ]", " ").split("\\s+");

						tweet = "";
						for (String tTok : tTokens) {
							if(tTok.length() > 1)
								tweet += tTok.trim() + " ";
						}
						
						length = tTokens.length;

						tTokensFirstHalf = Arrays.copyOfRange(tTokens, 0, length/2);
						tTokensSecondHalf = Arrays.copyOfRange(tTokens, length/2, length);


						for (String tTok : tTokensFirstHalf) {
							if(tTok.length() > 1)
								firstTweet += tTok.trim() + " ";
						}

						for (String tTok : tTokensSecondHalf) {
							if(tTok.length() > 1)
								secondTweet += tTok.trim() + " ";
						}
						
						pw.write(ID + "\t" + tweet + "\n");
						pw1.write(ID + "\t" + firstTweet + "\n");
						pw2.write(ID + "\t" + secondTweet + "\n");
						
					}
					pw.close();
					pw1.close();
					pw2.close();
				} catch (FileNotFoundException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}
				
			}
		}
		
	}
	static void RunDemo()
	{
		try {
			GetTextBlob_LDAFiles(); //This is the first method to run to obtain the input files (tweetID-tweet tokens) for LDA and TextBlob processing.
			//Read SW3.txt and put the values in SentiWordNet.
			ReadSentiWordNet("StefanFiles_HCOMP/resources/SW3.txt");
			//Read Twitter-Lexicon.txt and put the values in TwitterLexicon.
			ReadTwitterLexicon("StefanFiles_HCOMP/resources/SemEval2015-English-Twitter-Lexicon.txt");

			//Read these files and put the values in the corresponding lists as PosWords, NegWords, StopWords, and LingoWords.
			PosWords = ReadWordList("StefanFiles_HCOMP/resources/posWords.txt");
			NegWords = ReadWordList("StefanFiles_HCOMP/resources/negWords.txt");
			StopWords = ReadWordList("StefanFiles_HCOMP/resources/stopWords.txt");
			LingoWords = ReadWordList("StefanFiles_HCOMP/resources/Chosen_Lingo.txt");

			File folder = new File("StefanFiles_HCOMP/dataset_splits/LA_Tweets_Percentage_2200/");
			//Get all the files in the folder to be processed (id-tab-tweet)
			File[] listOfFiles = folder.listFiles();

			String datasetFile = "";
			String filePart = "";
			String tfFile = "";

			String textBlobFile = "";
			String textBlobFile1 = "";
			String textBlobFile2 = "";
			
			String featureFile = "";
			
			
			//Two folder containing the rosette .json files for labelled and unlabelled datasets
			String rosetteFileUnlabelled = "StefanFiles_HCOMP/resources/rosette_sentiment_twitter_full_trump_unlabelled_25451" + "/";
			String rosetteFileLabelled = "StefanFiles_HCOMP/resources/rosette_sentiment_twitter_trump_labelled" + "/";
					
			//Get all the files in these folders
			File folderUnRos = new File(rosetteFileUnlabelled);
			List<File> listOfFilesUnRos =  Arrays.asList(folderUnRos.listFiles());
			
			File folderLabRos = new File(rosetteFileLabelled);
			List<File> listOfFilesLabRos = Arrays.asList(folderLabRos.listFiles());
			
			//This is the folder for the features files (FeatureFiles_2200)
			File folderFeature = new File("StefanFiles_HCOMP/feature_files/FeatureFiles_2200/FeatureFiles_2200_1/");
			List<File> listOfFeatureFiles = Arrays.asList(folderFeature.listFiles());
			
			//For each file in the dataset to be processed (on each line there is 1 instance as "id-tab-tweet")
			for (int i = 0; i < listOfFiles.length; i++) 
			{
				File currFile = listOfFiles[i];
				if (currFile.isFile()) 
				{		
					TweetLists = new ArrayList<Tweet>();
					Instances = new ArrayList<Instance>();
					
					datasetFile = listOfFiles[i].getName();
					
					//For other experiments, there are also files with .csv extension
					filePart = datasetFile.replace(".csv", "");
					filePart = filePart.replace(".txt", "");
					
					featureFile = "FeatureFile_Trump_" + filePart + ".txt";
					//If we haven't generates the feature file, yet then compute and print it. 
					if (FindFile(listOfFeatureFiles,featureFile) == false)
					{
						/*
						 * ----START HERE!!!-----
						 * Change these filenames for different experiments!!
						 */
						
						//Get the tf*idf scores for the current dataset file
						tfFile = "StefanFiles_HCOMP/tf_idf/tf_idf_2200/tf_idf_2200_1/" + filePart + ".json";
						
						//This file will be the output of LDA processing which contains topic distributions 
						//of the sentences in the current file
						String LDADoc = "StefanFiles_HCOMP/lda/LDADocs_2200/LDADocs_2200_1/From" + "LDA_Document_Trump_" + filePart + ".txt";

						//These files will be output files of TextBlob processing which contain polarity values for the
						//given part of each sentence in the current dataset file (
						textBlobFile = "StefanFiles_HCOMP/textBlob/TextBlob_2200/TextBlob_2200_1/TextBlob_" + "LDA_Document_Trump_" + filePart + ".txt";
						textBlobFile1 = "StefanFiles_HCOMP/textBlob/TextBlob_2200/TextBlob_2200_1/TextBlob_" + "LDA_DocumentFirst_Trump_" + filePart + ".txt";
						textBlobFile2 = "StefanFiles_HCOMP/textBlob/TextBlob_2200/TextBlob_2200_1/TextBlob_" + "LDA_DocumentSecond_Trump_" + filePart + ".txt";
						
						//Read the dataset file and put all the tweets in the list of "TweetLists".
						//ReadDataset("StefanFiles_HCOMP/dataset_splits/LA_Tweets_Percentage_2200/" + datasetFile);
						//Get all the tf-values from the file and put them to the "FrequencyScores".
						ReadFrequencyScores(tfFile);
					
						//If we think that this is a dataset of search query-returning documents as mentioned in the paper,
						//we have only one query (but many documents), thus there is no need to compute the frequency values over and over again for the
						//given one query.
						Calculate_FrequencyValuesQuery();

						JSONParser parser = new JSONParser();

						String ID = "";
						String tweet = "";


						double val = 0.0, val1 = 0.0, val2 = 0.0;
						int length = 0;

						List<Double> values = new ArrayList<Double>();

						int tweetSize = TweetLists.size();
						//For each tweet in the list, we will compute our features as described in a feature table of the paper.
						for (Tweet t : TweetLists) 
						{						
							ID = t.ID;

							tweet = t.text;

							int excMarks = 0;
							int questMarks = 0;
							int posEmots = 0;
							int negEmots = 0;
							int suspPoints = 0;

							int numAllUppercaseTokens = 0;
							int numRepeatingCharsTokens = 0;

							tweet = tweet.replaceAll("http://[^\\s]+", "");

							Instance ins = new Instance();	
							//If there is a retweet in the given tweet or not (0, or 1)
							ins.retweet = FindRT_Token(tweet);
							ins.key = ID;

							Object obj = null;
							try {

								//Set the query values (they are same for all the tweets in the list!)
								//Because we have only one query for the whole dataset!!
								//Now I don't use the query features into the feature list since they are all same for all the tweets in the list.
								//But if there are more than one query-task then you can use these values in your feature-vector, as well.
								
/*								ins.qsumFreq = qsumFreq;
								ins.qmeanFreq = qmeanFreq;
								ins.qminFreq = qminFreq;
								ins.qmaxFreq = qmaxFreq;
								ins.qvarianceFreq = qvarianceFreq;*/

								numAllUppercaseTokens = FindAllUpperCase_Token(tweet);
								ins.numAllUppercaseTokens = numAllUppercaseTokens;
								sumAllUppercaseTokens += numAllUppercaseTokens;	

								tweet = RemoveStopWords(tweet);
								tweet = tweet.toLowerCase(Locale.ENGLISH);
								String[] tTokens = tweet.split("\\s+");

								//Get the list of all the feature values related to the TwitterLexicon.
								values = FindTwitterLexicon_Keyword(tweet);

								//Then obtain each value and assign it to the suitable place in the feature vector.
								ins.tminPolTokenTwitter = values.get(0);
								ins.tAvgPolTwitter = values.get(1);
								ins.tmaxPolTokenTwitter = values.get(2);

								ins.tFirstminPolTokenTwitter = values.get(3);
								ins.tFirstAvgPolTwitter = values.get(4);
								ins.tFirstmaxPolTokenTwitter = values.get(5);

								ins.tSecondminPolTokenTwitter = values.get(6);
								ins.tSecondAvgPolTwitter = values.get(7);
								ins.tSecondmaxPolTokenTwitter = values.get(8);

								sumAvgPol += ins.tAvgPolTwitter;
								sumAvgPolFirst += ins.tFirstAvgPolTwitter;
								sumAvgPolSecond += ins.tSecondAvgPolTwitter;

								//These are the punctuation-related features.
								excMarks = FindExclamationMarks(tweet);
								questMarks = FindQuestionMarks(tweet);
								posEmots = FindPosEmoticons(tweet);
								negEmots = FindNegEmoticons(tweet);
								suspPoints = FindSuspensionPoints(tweet);

								ins.tNumExcMarks = excMarks;
								ins.tNumQuestMarks = questMarks;
								ins.tNumPosEmot = posEmots;
								ins.tNumNegEmot = negEmots;
								ins.tNumSuspensionPoints = suspPoints;
								ins.numQuotationMarks = FindQuotationMarks(tweet);
								ins.numTwitterLingos = FindLingos(tweet);

								sumExcMarks += excMarks;
								sumQuestMarks += questMarks;
								sumPosEmots += posEmots;
								sumNegEmots += negEmots;
								sumSuspPoints += suspPoints;

								tTokens = tweet.replaceAll("[^a-zA-Z ]", " ").split("\\s+");
								ins.tTokens = tTokens;

								tweet = "";
								for (String tTok : tTokens) {
									if(tTok.length() > 1)
										tweet += tTok.trim() + " ";
								}
								tTokens = tweet.split("\\s+");
								ins.tTokens = tTokens;

								//These are the other features again described in the feature table.
								ins.numYet = CountKeywordYet(tTokens);
								ins.numSudden = CountKeywordSudden(tTokens);
								ins.tNumKeywordLike = CountKeywordLike(tTokens);
								ins.tNumKeywordWould = CountKeywordWould(tTokens);

								numRepeatingCharsTokens = FindRepeatingCharacters(tweet);
								ins.numRepeatingCharactersTokens = numRepeatingCharsTokens;
								sumRepeatingCharsTokens += numRepeatingCharsTokens;

								ins.tweet = tweet;
								length = tTokens.length;
								ins.tLength = length;

								ins.tTokensFirstHalf = Arrays.copyOfRange(tTokens, 0, length/2);
								ins.tTokensSecondHalf = Arrays.copyOfRange(tTokens, length/2, length);
								
								//Now, these are the outputs of the TextBlob processing.
								//The files contain the sentiment values for the whole, first part, and second part of the text.
								try {
									ReadTextBlobFile(textBlobFile);
									ReadTextBlobFirstFile(textBlobFile1);
									ReadTextBlobSecondFile(textBlobFile2);
								} catch (Exception e) {
									// TODO Auto-generated catch block
									e.printStackTrace();
								}					

								val = GetBlobSentiment(ins.key);
								sumTextBlobSentiment += val;		
								ins.textBlobSentiment = val;

								val1 = GetBlobSentimentFirst(ins.key);
								sumTextBlobSentimentFirst += val1;		
								ins.textBlobSentimentFirst = val1;

								val2 = GetBlobSentimentSecond(ins.key);
								sumTextBlobSentimentSecond += val2;		
								ins.textBlobSentimentSecond = val2;

								//Compute SentiWordNet related features (similar to the Twitter Lexicon).
								Compute_SWN_Scores(ins);

								//Parse the rosette .json files for the labelled dataset.
								for (File f : listOfFilesLabRos) {
									String fName = f.getName();
									if(fName.equals(ID + ".json"))
									{
										obj = parser.parse(new FileReader(rosetteFileLabelled + "/" + ID + ".json"));
										break;
									}
								}
								
								//Parse the rosette .json files for the unlabelled dataset.
								if(obj ==  null)
								{
									for (File f : listOfFilesUnRos) {
										String fName = f.getName();
										if(fName.equals(ID + ".json"))
										{
											obj = parser.parse(new FileReader(rosetteFileUnlabelled + "/" + ID + ".json"));
											break;
										}
									}
								}


								//Then obtain POSTag information from the parsed rosette files.
								//Count the number of POSTags and put the information in the feature-vector.
								if(obj != null)
								{
									JSONObject jsonObject = (JSONObject) obj;
									JSONObject obj1 = (JSONObject) jsonObject.get("POS");

									if(obj1 != null)
									{
										JSONArray arr = (JSONArray)obj1.get("posTags");
										int tweetTokSize = arr.size();
										for (int a = 0; a < tweetTokSize; a++)
										{
											String posTag = (String) arr.get(a);
											if(posTag != null)
											{
												if(posTag.equals("PROPN"))
													ins.tNumNouns++;
												else if(posTag.equals("ADJ"))
													ins.tNumAdjectives++;
												else if(posTag.equals("ADV"))
													ins.tNumAdverbs++;
												else if(posTag.equals("VERB"))
													ins.tNumVerbs++;
											}
										}

										ins.nounRatio = ins.tNumNouns / ins.tLength;
										ins.adjRatio = ins.tNumAdjectives / ins.tLength;
										ins.advRatio = ins.tNumAdverbs / ins.tLength;
										ins.verbRatio = ins.tNumVerbs / ins.tLength;
									}
								}


								/*String query = t.query;
								query = query.replaceAll("http://[^\\s]+", "");

								query = query.toLowerCase(Locale.ENGLISH);
								query = RemoveStopWords(query);

								ins.qAvgPolTwitter = FindTwitterLexicon_Keyword(query).get(1);

								String[] qTokens = query.replaceAll("[^a-zA-Z ]", "").split("\\s+");

								ins.qTokens = qTokens;
								query = "";
								for (String qTok : qTokens) {
									if(!qTok.equals(""))
										query += qTok.trim() + " ";
								}

								ins.query = query;
								ins.qLength = qTokens.length;*/


								//ins.numOfQOccurInTweet = FindQuery_InTweet(query, tweet);
								
								//These are the other features displayed in the feature table of the paper.
								ins.queryTermProx = ComputeQueryTerm_Proximity(query, ins.tTokens);
								ins.levenshteinDistance = LevenshteinCompute(query, tweet);
								ins.jaccardSimilarityValues = JaccardSimilarityCompute(query, tweet);

								ins.tPosWords = FindPosKeywords(ins.tTokens);
								ins.tNegWords = FindNegKeywords(ins.tTokens);

								ins.tPosWordsFirst = FindPosKeywords(ins.tTokensFirstHalf);
								ins.tNegWordsFirst = FindNegKeywords(ins.tTokensFirstHalf);

								ins.tPosWordsSecond = FindPosKeywords(ins.tTokensSecondHalf);
								ins.tNegWordsSecond = FindNegKeywords(ins.tTokensSecondHalf);

								ins.tPosTermRatio = ins.tPosWords / ins.tLength;
								ins.tNegTermRatio = ins.tNegWords / ins.tLength;

								ins.tPosTermRatioFirst = ins.tPosWordsFirst / ins.tLength / 2;
								ins.tNegTermRatioFirst = ins.tNegWordsFirst / ins.tLength / 2;

								ins.tPosTermRatioSecond = ins.tPosWordsSecond / ins.tLength / 2;
								ins.tNegTermRatioSecond = ins.tNegWordsSecond / ins.tLength / 2;


								Calculate_FrequencyValuesTweet(ins);

								//We add the current instance (one tweet with its properties) to the list.
								Instances.add(ins);
							}

							catch (FileNotFoundException e1) {
								// TODO Auto-generated catch block
								e1.printStackTrace();
							} catch (IOException e1) {
								// TODO Auto-generated catch block
								e1.printStackTrace();
							} catch (ParseException e1) {
								// TODO Auto-generated catch block
								e1.printStackTrace();
							}


						}
						sumAvgPol = (sumAvgPol + 0.01 / tweetSize);
						sumAvgPolFirst = (sumAvgPolFirst + 0.01 / tweetSize / 2);
						sumAvgPolSecond = (sumAvgPolSecond + 0.01 / tweetSize / 2);

						sumAvgPolSenti = (sumAvgPolSenti + 0.01 / tweetSize);
						sumAvgPolSentiFirst = (sumAvgPolSentiFirst + 0.01 / tweetSize / 2);
						sumAvgPolSentiSecond = (sumAvgPolSentiSecond + 0.01/ tweetSize / 2);

						sumDomPolSenti = (sumDomPolSenti + 0.01 / tweetSize);
						sumDomPolSentiFirst = (sumDomPolSentiFirst + 0.01 / tweetSize / 2);
						sumDomPolSentiSecond = (sumDomPolSentiSecond + 0.01 / tweetSize / 2);

						sumTextBlobSentiment = (sumTextBlobSentiment+ 0.01  / tweetSize);
						sumTextBlobSentimentFirst = (sumTextBlobSentimentFirst + 0.01 / tweetSize / 2);
						sumTextBlobSentimentSecond = (sumTextBlobSentimentSecond + 0.01 / tweetSize / 2);

						PrintWriter pw3 = null;
						String fData = "";
						try {
							//Get the topic distribution for each tweet in the current dataset file.
							ReadDocRepresentationsLDA(LDADoc);
							pw3 = new PrintWriter("StefanFiles_HCOMP/feature_files/FeatureFiles_2200/FeatureFiles_2200_1/" + featureFile);
							//PrintWriter pw2 = new PrintWriter("LDA.txt");
							int insIndex = 0;
							//For each instance in the "Instances" list (each instance -> each tweet in the current dataset file, which will be a feature vector) 
							for (Instance ins : Instances) 
							{
								ins.excRatio = ins.tNumExcMarks / ((double) sumExcMarks + 0.01);
								ins.questRatio = ins.tNumQuestMarks / ((double) sumQuestMarks + 0.01);
								ins.posEmotRatio = ins.tNumPosEmot / ((double)  sumPosEmots + 0.01);
								ins.negEmotRatio = ins.tNumNegEmot / ((double) sumNegEmots + 0.01);

								ins.tavgPolRatioTwitter = ins.tAvgPolTwitter / (sumAvgPol/ + 0.01);
								ins.tAvgPolRatioSenti = ins.tAvgPolSenti / (sumAvgPolSenti + 0.01);
								ins.tDomPolRatioSenti = ins.tDomPolSenti / (sumDomPolSenti + 0.01);
								ins.textBlobSentimentRatio = ins.textBlobSentiment / (sumTextBlobSentiment + 0.01);

								ins.tFirstavgPolRatioTwitter = ins.tFirstAvgPolTwitter / (sumAvgPolFirst / + 0.01);
								ins.tFirstAvgPolRatioSenti = ins.tFirstAvgPolSenti / (sumAvgPolSentiFirst + 0.01);
								ins.tFirstDomPolRatioSenti = ins.tFirstDomPolSenti / (sumDomPolSentiFirst + 0.01);
								ins.textBlobSentimentRatioFirst = ins.textBlobSentimentFirst / (sumTextBlobSentimentFirst + 0.01);

								ins.tSecondavgPolRatioTwitter = ins.tSecondAvgPolTwitter / (sumAvgPolSecond / + 0.01);
								ins.tSecondAvgPolRatioSenti = ins.tSecondAvgPolSenti / (sumAvgPolSentiSecond + 0.01);
								ins.tSecondDomPolRatioSenti = ins.tSecondDomPolSenti / (sumDomPolSentiSecond + 0.01);
								ins.textBlobSentimentRatioSecond = ins.textBlobSentimentSecond / (sumTextBlobSentimentSecond + 0.01);

								ins.numAllUppercaseToken_Ratio = ins.numAllUppercaseTokens / ((double) sumAllUppercaseTokens + 0.01);
								ins.numRepeatingCharactersTokens_Ratio = ins.numRepeatingCharactersTokens / ((double) sumRepeatingCharsTokens + 0.01);

								//This is the feature file content data, we concatenate each feature vector to it and each feature is separated by comma (csv file)
								fData += ins.key + "\t" + ins.tAvgPolTwitter + "," + ins.tavgPolRatioTwitter + "," + ins.tminPolTokenTwitter + "," + ins.tmaxPolTokenTwitter + 
										"," + ins.tFirstAvgPolTwitter + "," + ins.tFirstavgPolRatioTwitter + "," + ins.tFirstminPolTokenTwitter + 
										"," + ins.tFirstmaxPolTokenTwitter + "," + ins.tSecondAvgPolTwitter + "," + ins.tSecondavgPolRatioTwitter + 
										"," + ins.tSecondminPolTokenTwitter + "," + ins.tSecondmaxPolTokenTwitter + "," + ins.tAvgPolSenti + 
										"," + ins.tAvgPolRatioSenti + "," + ins.tDomPolSenti + "," + ins.tDomPolRatioSenti + "," + ins.tminPolTokenSenti + 
										"," + ins.tmaxPolTokenSenti + "," + ins.tFirstAvgPolSenti + "," + ins.tFirstAvgPolRatioSenti + "," + ins.tFirstDomPolSenti + 
										"," + ins.tFirstDomPolRatioSenti + "," + ins.tSecondAvgPolSenti + "," + ins.tSecondAvgPolRatioSenti + 
										"," + ins.tSecondDomPolSenti + "," + ins.tSecondDomPolSenti + "," + ins.textBlobSentiment + "," + ins.textBlobSentimentRatio + 
										"," + ins.textBlobSentimentFirst + "," + ins.textBlobSentimentRatioFirst + "," + ins.textBlobSentimentSecond + 
										"," + ins.textBlobSentimentRatioSecond + "," + ins.tLength + "," + ins.retweet + "," + ins.tPosWords + 
										"," + ins.tNegWords + "," + ins.tPosTermRatio + "," + ins.tNegTermRatio + "," + ins.tLengthFirst + "," + ins.tLengthSecond + 
										"," + ins.tPosWordsFirst + "," + ins.tPosTermRatioFirst + "," + ins.tNegWordsFirst + "," + ins.tNegTermRatioFirst +
										"," + ins.tPosWordsSecond + "," + ins.tPosTermRatioSecond + "," + ins.tNegWordsSecond + "," + ins.tNegTermRatioSecond +
										"," + ins.tsumFreq + "," + ins.tmeanFreq + "," + ins.tminFreq + "," + ins.tmaxFreq + "," + ins.tvarianceFreq + 
										"," + ins.tNumNouns + "," + ins.tNumAdjectives + "," + ins.tNumAdverbs + "," + ins.tNumVerbs + 
										"," + ins.nounRatio + "," + ins.adjRatio + "," + ins.advRatio + "," + ins.verbRatio + "," + ins.queryTermProx + 
										"," + ins.tNumExcMarks + "," + ins.tNumQuestMarks + "," + ins.tNumSuspensionPoints +
										"," + ins.numQuotationMarks + "," + ins.tNumKeywordWould + "," + ins.tNumKeywordLike +
										"," + ins.numSudden + "," + ins.numYet + "," + ins.numTwitterLingos + 
										"," + ins.tNumPosEmot + "," + ins.tNumNegEmot + "," + ins.excRatio + "," + ins.questRatio + 
										"," + ins.posEmotRatio + "," + ins.negEmotRatio + "," + ins.numAllUppercaseTokens + 
										"," + ins.numAllUppercaseToken_Ratio + "," + ins.numRepeatingCharactersTokens +
										"," + ins.numRepeatingCharactersTokens_Ratio + "," + ins.levenshteinDistance + 
										"," + ins.jaccardSimilarityValues.get(0) + "," + ins.jaccardSimilarityValues.get(1) + ",";

								//Get the topic distribution by giving the index of the current instance.
								List<Double> DocDist = new ArrayList<Double>();
								DocDist = TopicProbabilities.get(insIndex);

								//We represent each tweet with 10 documents but if any of the topics do not exist in the distribution, we handle this.
								//In order to have a same sized topic distibution vector and add it to the feature vecotr of the current instance.
								if(DocDist.size() > 0)
								{
									for (int j = 0; j < DocDist.size()-1; j++) {
										fData += String.valueOf(DocDist.get(j)) + ",";

									}

									fData += String.valueOf(DocDist.get(DocDist.size()-1));
									fData += "\n";
								}
								insIndex++;
							}
							pw3.write(fData);
							pw3.close();

						} catch (IOException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						} 
						catch (ParseException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}
					else
						System.out.println(featureFile);
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (ParseException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	//Read the given dataset file.
	public static void ReadDataset(String filename)
	{
		try {
			File f = new File(filename);

			BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";
			/*			while ((line = b.readLine()) != null) {
				lineCount++;
			}*/
			Tweet t = new Tweet();
			//b.readLine();
			while ((line = b.readLine()) != null) 
			{

				t = new Tweet();
				String [] obj = line.split("\t");
				t.ID = obj[0];
				t.text = obj[1];
				//t.qID = obj[3];
				//t.query = t.qID;
				//t.relevanceGrad = obj[4];

				TweetLists.add(t);
			}
			b.close();
		} catch (Exception e) {
			// TODO: handle exceptiontAvgPolTwitter
		}
	}
	
	//Counts the number of repeating characters.
	public static int FindRepeatingCharacters(String text)
	{
		int count = 0;
		String [] tokens = text.split("\\s+");
		for (String tok : tokens) {
			for(int i=0; i<tok.length()-1; i++)
			{
				if(tok.charAt(i) == tok.charAt(i+1))
				{
					if(i+2 < tok.length()-1)
					{
						if(tok.charAt(i) == tok.charAt(i+2))
						{
							count++;
							continue;
						}
					}
				}
			}
		}
		return count;
	}

	//Counts the number of occurrences of "yet" keyword.
	public static int CountKeywordYet(String [] tokens)
	{
		int count = 0;
		for (String str : tokens) {
			if(str.equals("yet"))
				count++;
		}
		return count;
	}

	//Counts the number of occurrences of "sudden" keyword.
	public static int CountKeywordSudden(String [] tokens)
	{
		int count = 0;
		for (String str : tokens) {
			if(str.equals("sudden"))
				count++;
		}
		return count;
	}
	
	//Counts the number of occurrences of "like" keyword.
	public static int CountKeywordLike(String [] tokens)
	{
		int count = 0;
		for (String str : tokens) {
			if(str.equals("like"))
				count++;
		}
		return count;
	}
	
	//Counts the number of occurrences of "would" keyword.
	public static int CountKeywordWould(String [] tokens)
	{
		int count = 0;
		for (String str : tokens) {
			if(str.equals("would"))
				count++;
		}
		return count;
	}

	//Counts the number of all-uppercased tokens.
	public static int FindAllUpperCase_Token(String text)
	{
		int count = 0;
		String [] tokens = text.split("\\s+");
		for (String tok : tokens) 
		{
			if(IsUpperCase(tok))
				count++;
		}
		return count;
	}

	//Checks if the given token contains an upper-cased character.
	private static boolean IsUpperCase(String token)
	{
		for (int i=0; i<token.length(); i++)
		{
			if (Character.isLowerCase(token.charAt(i)))
			{
				return false;
			}
		}
		return true;
	}

	//Counts the number of suspension points.
	public static int FindSuspensionPoints(String text)
	{
		Pattern pattern = Pattern.compile("\\.\\.");
		Matcher matcher = pattern.matcher(text);
		int count = 0;
		while (matcher.find())
			count++;

		return count;

	}

	//Counts the number of positive emoticons.
	public static int FindPosEmoticons(String text)
	{
		Pattern pattern = Pattern.compile(":\\)|:\\(|:D|=\\)|:-\\)|=]");
		Matcher matcher = pattern.matcher(text);
		int count = 0;
		while (matcher.find())
			count++;

		return count;
	}

	//Counts the number of negative emoticons.
	public static int FindNegEmoticons(String text)
	{
		Pattern pattern = Pattern.compile(":\\(|:-\\(|=\\(|:/|;\\)|;-\\)");
		Matcher matcher = pattern.matcher(text);
		int count = 0;
		while (matcher.find())
			count++;

		return count;
	}

	//Counts the number of lingo words in the LingoWords list.
	public static int FindLingos(String text)
	{
		int count = 0;
		Pattern pattern = null, pattern1 = null;
		Matcher matcher = null, matcher1 = null;
		for (String lingo : LingoWords) 
		{
			pattern = Pattern.compile("\\b"+lingo);
			pattern1 = Pattern.compile("\\b" + lingo + "\\b");
			matcher = pattern.matcher(text);
			matcher1 = pattern1.matcher(text);

			while (matcher.find())
			{
				if(matcher1.find())
					count++;
				else
					count = count+2;
			}			
		}
		return count;	
	}

	//Counts the number of quotation marks.
	public static int FindQuotationMarks(String text)
	{
		Pattern pattern = Pattern.compile("\".+\\s+.+\"");
		Matcher matcher = pattern.matcher(text);
		int count = 0;
		while (matcher.find())
			count++;

		return count;
	}

	//Counts the number of exclamation marks.
	public static int FindExclamationMarks(String text)
	{
		Pattern pattern = Pattern.compile("!");
		Matcher matcher = pattern.matcher(text);
		int count = 0;
		while (matcher.find())
			count++;

		return count;
	}

	//Counts the number of question marks.
	public static int FindQuestionMarks(String text)
	{
		Pattern pattern = Pattern.compile("\\?");
		Matcher matcher = pattern.matcher(text);
		int count = 0;
		while (matcher.find())
			count++;

		return count;
	}
	public static String removeUrl(String commentstr)
	{
		String urlPattern = "((https?|ftp|gopher|telnet|file|Unsure|http):((//)|(\\\\))+[\\w\\d:#@%/;$()~_?\\+-=\\\\\\.&]*)";
		Pattern p = Pattern.compile(urlPattern,Pattern.CASE_INSENSITIVE);
		Matcher m = p.matcher(commentstr);
		int i = 0;
		while (m.find()) {
			commentstr = commentstr.replaceAll(m.group(i),"").trim();
			i++;
		}
		return commentstr;
	}

	public static int FindRT_Token(String text)
	{
		String [] tokens = text.split("\\s+");
		for (String tok : tokens) {
			if(tok.equals("RT"))
				return 1;
		}
		return 0;
	}
	//Gets the topic distributions for all the tweets in the given dataset file.
	public static void ReadDocRepresentationsLDA(String filename) throws FileNotFoundException, IOException, ParseException
	{
		int numTopics = 10;
		TopicProbabilities = new ArrayList<List<Double>>(); 
		try 
		{
			File f = new File(filename);

			BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";

			while ((line = b.readLine()) != null) 
			{
				List<Double> Doc_TopicDistributions = new ArrayList<Double>();
				String [] obj = line.split(", ");

				String strVal = "";
				String topIndex = "";
				double ldaVal = 0.0;
				int prevIndex = 0, currInd = 0;
				for (int i = 0; i < obj.length-1; i=i+2) 
				{
					topIndex = obj[i].replaceAll("\\(", "");
					topIndex = topIndex.replaceAll("\\[", "");
					strVal = obj[i+1];
					strVal = strVal.replaceAll("\\)", "");
					strVal = strVal.replaceAll("]", "");
					ldaVal = Double.parseDouble(strVal);

					currInd = Integer.parseInt(topIndex);
					for(int j=prevIndex; j<currInd; j++)
					{
						Doc_TopicDistributions.add(0.0);
					}			
					Doc_TopicDistributions.add(ldaVal);
					prevIndex = currInd + 1;
				}
				if(Doc_TopicDistributions.size() < numTopics)
				{
					for (int k = prevIndex; k < numTopics; k++) {
						Doc_TopicDistributions.add(0.0);
					}
				}

				TopicProbabilities.add(Doc_TopicDistributions);
			}
			b.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	//Reads the twitter lexicon and store it into the TwitterLexicon.
	public static void ReadTwitterLexicon(String filename) throws FileNotFoundException, IOException, ParseException
	{
		try {

			File f = new File(filename);

			BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";

			while ((line = b.readLine()) != null) {
				String [] obj = line.split("\t");
				TwitterLexicon.put(obj[1], Double.parseDouble(obj[0]));
			}
			b.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	//Get the frequency scores from the .json files and put them to the "FrequencyScores" as (term-score) pairs.
	public static void ReadFrequencyScores(String filename) throws FileNotFoundException, IOException, ParseException
	{
		FrequencyScores = new LinkedHashMap<String, Double>();
		JSONParser parser = new JSONParser();

		Object obj;
		obj = parser.parse(new FileReader(filename));

		JSONObject jsonObject = (JSONObject) obj;

		for(Iterator iterator = jsonObject.entrySet().iterator(); iterator.hasNext();) 
		{
			Map.Entry<String, JsonElement> ent = (java.util.Map.Entry<String, JsonElement>) iterator.next();
			String term = ent.getKey();
			Object tfidf = ent.getValue();

			FrequencyScores.put(term, (Double) tfidf);
		}
	}

	//We've implemented this method to compute frequency scores of the tokens in the query. 
	//Yet, we only have one query in our task, so we will not use these features in our feature set.
	public static void Calculate_FrequencyValuesQuery()
	{
		List<Double> Scores = new ArrayList<Double>();

		double minFreq = 500.00;
		double maxFreq = -500.00;
		double sumFreq = 0.0;
		for (String tok : qTokens) {
			if(FrequencyScores.keySet().contains(tok))
			{
				double val = FrequencyScores.get(tok);
				sumFreq += val;

				Scores.add(val);

				if(val < minFreq)
					minFreq = val;
				if(val > maxFreq)
					maxFreq = val;
			}
		}

		qsumFreq = sumFreq;
		qmeanFreq = sumFreq / qTokens.length;
		qminFreq = minFreq;
		qmaxFreq = maxFreq;

		double varFreq = 0.0;
		for (Double d : Scores) {
			varFreq += Math.pow(d-qmeanFreq, 2);
		}
		//population-variance, N-1 is more common in the denominator
		qvarianceFreq = varFreq / (qTokens.length-1);
	}

	//Get the frequency scores from the "FrequencyScores" and make some computations for the tokens in each tweet.
	public static void Calculate_FrequencyValuesTweet(Instance ins)
	{
		List<Double> Scores = new ArrayList<Double>();

		double minFreq = 500.00;
		double maxFreq = -500.00;
		double sumFreq = 0.0;
		for (String tok : ins.tTokens) {
			if(FrequencyScores.keySet().contains(tok))
			{
				double val = FrequencyScores.get(tok);
				sumFreq += val;

				Scores.add(val);
				if(val < minFreq)
					minFreq = val;
				if(val > maxFreq)
					maxFreq = val;
			}
		}
		ins.tsumFreq = sumFreq;
		ins.tmeanFreq = sumFreq / ins.tLength;
		ins.tminFreq = minFreq;
		ins.tmaxFreq = maxFreq;

		double varFreq = 0.0;
		for (Double d : Scores) {
			varFreq += Math.pow(d-ins.tmeanFreq, 2);
		}
		ins.tvarianceFreq = varFreq / ins.tTokens.length;
	}

	//This method checks if the Twitter Lexicon contains the tokens in the given text and 
	//if so it obtains the polarity values from the lexicon.
	public static List<Double> FindTwitterLexicon_Keyword(String text)
	{
		List<Double> Values = new ArrayList<Double>();

		double maxPol = Double.MIN_VALUE;
		double minPol = Double.MAX_VALUE;

		double polToken = 0.0;
		double avgPol = 0.0;

		double polTokenFirst = 0.0;
		double maxPolFirst = Double.MIN_VALUE;
		double minPolFirst = Double.MAX_VALUE;	
		double avgPolFirst = 0.0;

		double polTokenSecond = 0.0;
		double maxPolSecond = Double.MIN_VALUE;
		double minPolSecond = Double.MAX_VALUE;
		double avgPolSecond = 0.0;

		//Tokenize the given text
		String [] tokens = text.split("\\s+");
		int size = tokens.length;

		//Get the tokens in the first part of the text.
		List<String> tokensFirst = Arrays.asList(Arrays.copyOfRange(tokens, 0, size/2));
		//Get the tokens in the second part of the text.
		List<String> tokensSecond = Arrays.asList(Arrays.copyOfRange(tokens, size/2, size));

		//For all the tokens in the given text, check if the TwitterLexicon contains it, or not.
		//If so, then get the polarity value for the corresponding token from the lexicon. 
		//We also store the max and min pol. value in the text (also for the first and second part of the text).
		for (String str : tokens) 
		{
			if(TwitterLexicon.containsKey(str))
			{		
				polToken = TwitterLexicon.get(str);
				avgPol += polToken;

				if(maxPol < polToken)
					maxPol = polToken;

				if(minPol > polToken)
					minPol = polToken;

				//Then do the same computation if the first part of the text contains the token.
				if(tokensFirst.contains(str))
				{
					polTokenFirst = polToken;
					avgPolFirst += polTokenFirst;

					if(maxPolFirst < polTokenFirst)
						maxPolFirst = polTokenFirst;

					if(minPolFirst > polTokenFirst)
						minPolFirst = polTokenFirst;
				}	

				//Then do the same computation if the second part of the text contains the token.
				if(tokensSecond.contains(str))
				{
					polTokenSecond = polToken;
					avgPolSecond += polTokenSecond;

					if(maxPolSecond < polTokenSecond)
						maxPolSecond = polTokenSecond;

					if(minPolSecond > polTokenSecond)
						minPolSecond = polTokenSecond;
				}
			}
		}
		//Compute avgPol for the given text.
		//Store also maxPol and minPol in the given text.
		avgPol = avgPol / size;
		avgPolFirst = avgPolFirst / size / 2;
		avgPolSecond = avgPolSecond / size / 2;

		Values.add(minPol);
		Values.add(avgPol);
		Values.add(maxPol);

		Values.add(minPolFirst);
		Values.add(avgPolFirst);
		Values.add(maxPolFirst);

		Values.add(minPolSecond);
		Values.add(avgPolSecond);
		Values.add(maxPolSecond);


		return Values;
	}
 
	//Counts the number of occurrences of the query in the given tweet (text matching).
	public static int FindQuery_InTweet(String q, String tweet)
	{
		int count = 0;
		int ind = tweet.indexOf(q);
		while(ind > 0)
		{
			count++;
			ind = tweet.indexOf(q, ind+q.length());
		}
		return count;
	}

	//Computes the query-term proximity feature as displayed in the feature table of the paper.
	public static int ComputeQueryTerm_Proximity(String q, String [] tokens)
	{
		int countTok = 0, maxCountTok = Integer.MIN_VALUE;
		List<String> tweetPartTokens = new ArrayList<String>();
		int tLength = tokens.length;
		if(wSize >= tLength)
		{
			tweetPartTokens = Arrays.asList(tokens);
			for (String tok : qTokens) {
				if(tweetPartTokens.contains(tok))
					countTok++;
			}
			maxCountTok = countTok;
		}
		else
		{
			for(int i=0; i+wSize<tLength; i++)
			{
				countTok = 0;
				tweetPartTokens = Arrays.asList(Arrays.copyOfRange(tokens, i, tLength));
				for (String tok : qTokens) 
				{
					if(tweetPartTokens.contains(tok))
						countTok++;
				}
				if(maxCountTok < countTok)
					maxCountTok = countTok;
			}
		}
		return maxCountTok;
	}

	public static String RemoveStopWords(String text)
	{
		String [] tokens = text.split("\\s+");
		for (String tok : tokens) {
			if(StopWords.contains(tok))
			{
				text = text.replaceAll("\\b"+tok+"\\b", "");
			}
		}
		return text;
	}
	
	//Finds positive keywords of the positive word list in the given list of tokens.
	public static int FindPosKeywords(String [] tokens)
	{
		int numPos = 0;

		for (String tok : tokens) {
			if(PosWords.contains(tok))
			{
				numPos++;
			}
		}
		return numPos;
	}
	//Finds negative keywords of the positive word list in the given list of tokens.
	public static int FindNegKeywords(String [] tokens)
	{
		int numNeg = 0;

		for (String tok : tokens) {
			if(NegWords.contains(tok))
			{
				numNeg++;
			}
		}
		return numNeg;
	}

	//After creating the shingles in other methods, this function computes jaccard similarity.
	public static List<Double> JaccardSimilarityCompute(String query, String tweet)
	{
		List<Double> jaccardSimilarities = new ArrayList<Double>();
		//String [] qTokens = query.split("//s+");
		for (int i = 1; i <= 2; i++) 
		{
			List<String> queryShingles = CreateShingles(query, i);
			List<String> tweetShingles = CreateShingles(tweet, i);

			double sharedShingles = 0;
			double unsharedShingles = 0;
			for (String shing : queryShingles) 
			{
				if(tweetShingles.contains(shing))
				{
					sharedShingles++;
				}
			}
			unsharedShingles = queryShingles.size() + tweetShingles.size() - sharedShingles;

			double jaccardSim = sharedShingles / unsharedShingles;
			jaccardSimilarities.add(jaccardSim);
		}
		return jaccardSimilarities;
	}

	public static List<String> CreateShingles(String text, int k)
	{
		List<String> Shingles = new ArrayList<String>();

		String [] tokens = text.split("\\s+"); 
		int shinglesNumber = tokens.length - k; 

		//Create all shingles 

		for (int i = 0; i <= shinglesNumber; i++) { 
			String shingle = ""; 

			//Create one shingle 
			for (int j = 0; j < k; j++) { 
				shingle = shingle + tokens[i+j]; 
				shingle += " ";
			} 

			shingle = shingle.trim();
			Shingles.add(shingle);
		} 

		return Shingles;
	}

	public static int LevenshteinCompute(String a, String b) {
		a = a.toLowerCase();
		b = b.toLowerCase();
		// i == 0
		int [] costs = new int [b.length() + 1];
		for (int j = 0; j < costs.length; j++)
			costs[j] = j;
		for (int i = 1; i <= a.length(); i++) {
			// j == 0; nw = lev(i - 1, j)
			costs[0] = i;
			int nw = i - 1;
			for (int j = 1; j <= b.length(); j++) {
				int cj = Math.min(1 + Math.min(costs[j], costs[j - 1]), a.charAt(i - 1) == b.charAt(j - 1) ? nw : nw + 1);
				nw = costs[j];
				costs[j] = cj;
			}
		}
		return costs[b.length()];
	}

	public static void ReadSentiWordNet(String filename)
	{
		try {

			SentiToken myTok = null;
			File f = new File(filename);

			BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";

			b.readLine();
			String term = "", POSTag = "", ID = "";

			while ((line = b.readLine()) != null) {
				myTok = new SentiToken();
				String [] obj = line.split("\t");

				POSTag = obj[0];
				term = obj[4].replaceAll("#.*", "");		
				//ID = term + "-" + POSTag;
				ID = term;

				if(SentiWordNet.containsKey(ID))
				{
					SentiWordNet.get(ID).posScoreList.add(Double.parseDouble(obj[2]));
					SentiWordNet.get(ID).negScoreList.add(-1*Double.parseDouble(obj[3]));
				}
				else
				{
					myTok.ID = ID;
					myTok.POSTag = POSTag;
					myTok.term = term;
					myTok.posScoreList.add(Double.parseDouble(obj[2]));
					myTok.negScoreList.add(-1*Double.parseDouble(obj[3]));

					SentiWordNet.put(ID, myTok);
				}
			}
			b.close();
			Process_SentiWordNet();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	//Computes the average and dominant polarities of the tokens in the SentiWordNet.
	//Also, it handles the synsets (if a token appears in different contexts, we aggregate these polarity values instead of taking into
	//account of the POSTag and so on.
	public static void Process_SentiWordNet()
	{
		double posScore = 0.0, negScore = 0.0;
		String ID = "";
		List<Double> PosScores = new ArrayList<Double>();
		List<Double> NegScores = new ArrayList<Double>();

		SentiToken ST = new SentiToken();
		for (java.util.Map.Entry<String, SentiToken> ent: SentiWordNet.entrySet()) 
		{			
			ST = ent.getValue();
			ID = ST.ID;

			PosScores = ST.posScoreList;
			if(PosScores.size() > 1)
			{
				for (double val : PosScores) {
					posScore += val;
				}
			}
			posScore = posScore/PosScores.size();

			NegScores = ent.getValue().negScoreList;
			if(NegScores.size() > 1)
			{
				for (double val : NegScores) {
					negScore += val;
				}
			}
			negScore = negScore/NegScores.size();

			ST.avgPol = (posScore + negScore) / 2;
			if(posScore > (-1*negScore))
			{
				ST.domPol = posScore;
			}
			else
			{
				ST.domPol = negScore;
			}
			ST.posPol = posScore;
			ST.negPol = negScore;

			//System.out.println(SentiWordNet.get(ID).term);
			SentiWordNet.put(ID, ST); //replace the value if it already exists

			//SentiWordNet.replace(ID, ST);
		}
	}

	//Bugs in sumAvgPolSenti computation have been corrected! - 22.01.2018
	//Obtains the polarity values of the tokens in the given text from the SentiWordNet.
	public static void Compute_SWN_Scores(Instance myIns)
	{
		List<String> First = Arrays.asList(myIns.tTokensFirstHalf);
		double tokAvgPol = 0.0, tokDomPol = 0.0;

		double avgPol = 0.0, domPol = 0.0, maxPol = Double.MIN_VALUE, minPol = Double.MAX_VALUE;
		double avgPolFirst = 0.0, domPolFirst = 0.0, maxPolFirst = Double.MIN_VALUE, minPolFirst = Double.MAX_VALUE;
		double avgPolSecond = 0.0, domPolSecond = 0.0, maxPolSecond = Double.MIN_VALUE, minPolSecond = Double.MAX_VALUE;

		SentiToken ST = new SentiToken();

		int tokSize = myIns.tTokens.length;

		for (String str : myIns.tTokens) 
		{
			if(SentiWordNet.containsKey(str))
			{
				ST = SentiWordNet.get(str);
				tokAvgPol = ST.avgPol;
				tokDomPol = ST.domPol;

				avgPol += tokAvgPol;
				domPol += tokDomPol;

				if(maxPol < avgPol)
					maxPol = avgPol;
				if(minPol > avgPol)
					minPol = avgPol;

				if(First.contains(str))
				{
					avgPolFirst += tokAvgPol;
					domPolFirst += tokDomPol;

					if(maxPolFirst < avgPolFirst)
						maxPolFirst = avgPolFirst;
					if(minPolFirst > avgPolFirst)
						minPolFirst = avgPolFirst;
				}
				else
				{
					avgPolSecond += tokAvgPol;
					domPolSecond += tokDomPol;

					if(maxPolSecond < avgPolSecond)
						maxPolSecond = avgPolSecond;
					if(minPolSecond > avgPolSecond)
						minPolSecond = avgPolSecond;
				}

			}
		}
		//Add avgPol & domPol of the tweet to the sum of the tweet list!
		sumAvgPolSenti += avgPol;
		sumDomPolSenti += domPol;

		sumAvgPolSentiFirst += avgPolFirst;
		sumDomPolSentiFirst += domPolFirst;

		sumAvgPolSentiSecond += avgPolSecond;
		sumDomPolSentiSecond += domPolSecond;

		myIns.tAvgPolSenti = avgPol / tokSize;
		myIns.tDomPolSenti = domPol / tokSize;
		myIns.tmaxPolTokenSenti = maxPol;
		myIns.tminPolTokenSenti = minPol;

		myIns.tFirstAvgPolSenti = avgPolFirst / tokSize/2;
		myIns.tFirstDomPolSenti = domPolFirst / tokSize/2;
		myIns.tFirstmaxPolTokenSenti = maxPolFirst;
		myIns.tFirstminPolTokenSenti = minPolFirst;

		myIns.tSecondAvgPolSenti = avgPolSecond / tokSize/2;
		myIns.tSecondDomPolSenti = domPolSecond / tokSize/2;
		myIns.tSecondmaxPolTokenSenti = maxPolSecond;
		myIns.tSecondminPolTokenSenti = minPolSecond;
	}

	public static void ReadTextBlobFile(String filename)
	{
		TextBlobOverallSentimentScores = new LinkedHashMap<String, BlobInstance>();
		try {

			File f = new File(filename);

			BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";

			BlobInstance BI = new BlobInstance();
			while ((line = b.readLine()) != null) {

				BI = new BlobInstance();
				String [] obj = line.split("\t");
				BI.ID = obj[0].trim();
				BI.tweet = obj[1];

				BI.blobSentiment = Double.parseDouble(obj[2]);

				TextBlobOverallSentimentScores.put(obj[0].trim(), BI);
			}
			b.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void ReadTextBlobFirstFile(String filename)
	{
		TextBlobOverallSentimentScoresFirst = new LinkedHashMap<String, BlobInstance>();
		try {

			File f = new File(filename);

			BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";

			BlobInstance BI = new BlobInstance();
			while ((line = b.readLine()) != null) {

				BI = new BlobInstance();
				String [] obj = line.split("\t");
				BI.ID = obj[0].trim();
				BI.tweet = obj[1];

				BI.blobSentiment = Double.parseDouble(obj[2]);

				TextBlobOverallSentimentScoresFirst.put(obj[0].trim(), BI);
			}
			b.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static void ReadTextBlobSecondFile(String filename)
	{
		TextBlobOverallSentimentScoresSecond = new LinkedHashMap<String, BlobInstance>();
		try {

			File f = new File(filename);

			BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";

			BlobInstance BI = new BlobInstance();
			while ((line = b.readLine()) != null) {

				BI = new BlobInstance();
				String [] obj = line.split("\t");
				BI.ID = obj[0].trim();
				BI.tweet = obj[1];

				BI.blobSentiment = Double.parseDouble(obj[2]);

				TextBlobOverallSentimentScoresSecond.put(obj[0].trim(), BI);
			}
			b.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	public static double GetBlobSentiment(String ID)
	{
		double val = 0.0;
		if(TextBlobOverallSentimentScores.keySet().contains(ID))
		{
			BlobInstance BI = TextBlobOverallSentimentScores.get(ID);
			val = BI.blobSentiment;
		}
		return val;
	}

	public static double GetBlobSentimentFirst(String ID)
	{
		double val = 0.0;
		if(TextBlobOverallSentimentScoresFirst.keySet().contains(ID))
		{
			BlobInstance BI = TextBlobOverallSentimentScoresFirst.get(ID);
			val = BI.blobSentiment;
		}
		return val;
	}

	public static double GetBlobSentimentSecond(String ID)
	{
		double val = 0.0;
		if(TextBlobOverallSentimentScoresSecond.keySet().contains(ID))
		{
			BlobInstance BI = TextBlobOverallSentimentScoresSecond.get(ID);
			val = BI.blobSentiment;
		}
		return val;
	}

	//Reads the word lists.
	//The function is used to read the seed word lists, e.g. positive, negative words.
	public static List<String> ReadWordList(String filename)
	{
		List<String> Words = new ArrayList<String>();
		try {

			File f = new File(filename);

			BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";


			while ((line = b.readLine()) != null) {
				Words.add(line);
			}
			b.close();

		} catch (IOException e) {
			e.printStackTrace();
		}
		return Words;
	}

	static boolean FindFile(List<File> listOfFeatureFiles, String filename)
	{
		for (File myFile : listOfFeatureFiles) 
		{
			String path;
			path = myFile.getName();
			if(path.contains(filename))
				return true;
		}
		return false;
	}
}