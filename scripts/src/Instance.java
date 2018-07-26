import java.util.ArrayList;
import java.util.List;


public class Instance {
	/*query
	 */

	String key;
/*	String [] qTokens;*/
/*	String query;
	double qLength;
	int qPosWords = 0;
	int qNegWords = 0;
	double qPosTermRatio = 0.0;
	double qNegTermRatio = 0.0;*/
	
	double qsumFreq;
	double qmeanFreq;
	double qminFreq;
	double qmaxFreq;
	double qvarianceFreq;
	
	//double qAvgPolTwitter;
	
	/*document
	 */
	
	String [] tTokens;
	String [] tTokensFirstHalf;
	String [] tTokensSecondHalf;
	String tweet;
	
	int numYet = 0;
	int numSudden = 0;
	
	double tLength;
	double tLengthFirst;
	double tLengthSecond;
	
	
	int tPosWords = 0;
	int tNegWords = 0;
	double tPosTermRatio = 0.0;
	double tNegTermRatio = 0.0;
	
	int tPosWordsFirst = 0;
	int tNegWordsFirst = 0;
	double tPosTermRatioFirst = 0.0;
	double tNegTermRatioFirst = 0.0;
	
	int tPosWordsSecond = 0;
	int tNegWordsSecond = 0;
	double tPosTermRatioSecond = 0.0;
	double tNegTermRatioSecond = 0.0;
	
	double avgPol = 0.0;
	
	int tNumNouns = 0;
	int tNumAdjectives = 0;
	int tNumAdverbs = 0;
	int tNumVerbs = 0;
	
	double nounRatio = 0.0;
	double adjRatio = 0.0;
	double advRatio = 0.0;
	double verbRatio = 0.0;
	
	double tsumFreq;
	double tmeanFreq;
	double tminFreq;
	double tmaxFreq;
	double tvarianceFreq;
	
	double tAvgPolSenti;
	double tDomPolSenti;
	double tmaxPolTokenSenti;
	double tminPolTokenSenti;
	
	double tAvgPolRatioSenti;
	double tDomPolRatioSenti;
	
	double textBlobSentiment;
	double textBlobSentimentRatio;
	
	double tAvgPolTwitter;
	double tavgPolRatioTwitter;
	double tmaxPolTokenTwitter;
	double tminPolTokenTwitter;
	
	
	//first half sentiment scores
	double tFirstAvgPolSenti;
	double tFirstDomPolSenti;
	double tFirstmaxPolTokenSenti;
	double tFirstminPolTokenSenti;
	
	double tFirstAvgPolRatioSenti;
	double tFirstDomPolRatioSenti;
	
	double textBlobSentimentFirst;
	double textBlobSentimentRatioFirst;
	
	double tFirstAvgPolTwitter;
	double tFirstavgPolRatioTwitter;
	double tFirstmaxPolTokenTwitter;
	double tFirstminPolTokenTwitter;
	
	//second half sentiment scores
	double tSecondAvgPolSenti;
	double tSecondDomPolSenti;
	double tSecondmaxPolTokenSenti;
	double tSecondminPolTokenSenti;
	
	double tSecondAvgPolRatioSenti;
	double tSecondDomPolRatioSenti;
	
	double textBlobSentimentSecond;
	double textBlobSentimentRatioSecond;
	
	double tSecondAvgPolTwitter;
	double tSecondavgPolRatioTwitter;
	double tSecondmaxPolTokenTwitter;
	double tSecondminPolTokenTwitter;

	int retweet = 0;
	
	int tNumKeywordLike;
	int tNumKeywordWould;
	
	int tNumExcMarks;
	int tNumQuestMarks;
	int tNumPosEmot;
	int tNumNegEmot;
	int tNumSuspensionPoints;
	int numQuotationMarks;
	int numTwitterLingos;
	
	double excRatio;
	double questRatio;
	double posEmotRatio;
	double negEmotRatio;

	
	int numAllUppercaseTokens;
	double numAllUppercaseToken_Ratio;
	
	int numRepeatingCharactersTokens;
	double numRepeatingCharactersTokens_Ratio;
	
	/*both
	 * 
	 */
	//int numOfQOccurInTweet = 0;
	int queryTermProx = 0;
	
/*	double lengthRatio = 0.0;
	double posCompRatio = 0.0;
	double negCompRatio = 0.0;
	double avgPolTwitterRatio;*/
	
	double levenshteinDistance;
	List<Double> jaccardSimilarityValues; //different shingles
	
	List<Double> TopicDistributions;
}

