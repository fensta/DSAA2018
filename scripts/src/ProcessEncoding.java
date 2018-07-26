import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.UnsupportedEncodingException;
import java.io.Writer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Random;

public class ProcessEncoding {

	public static LinkedHashMap<Integer, Batch_Ins> LA_Dataset = new LinkedHashMap<Integer, Batch_Ins>();
	public static LinkedHashMap<Integer, Batch_Ins> NotLA_Dataset = new LinkedHashMap<Integer, Batch_Ins>();
	
	public static LinkedHashMap<Integer, Batch_Ins> LA_DatasetStefan = new LinkedHashMap<Integer, Batch_Ins>();
	public static LinkedHashMap<Integer, Batch_Ins> NotLA_DatasetStefan = new LinkedHashMap<Integer, Batch_Ins>();
	
	public static LinkedHashMap<Integer, Batch_Ins> LA_DatasetMerged = new LinkedHashMap<Integer, Batch_Ins>();
	public static LinkedHashMap<Integer, Batch_Ins> NotLA_DatasetMerged = new LinkedHashMap<Integer, Batch_Ins>();
	
	public static Random randomGenerator = new Random();  
	
	
	public static void main(String[] args) throws FileNotFoundException, UnsupportedEncodingException {
		// TODO Auto-generated method stub
		
		//DoExperiment1or2(); //for experiment1 and 2
		DoExperiment3or4();
		
		

	}
	
	public static void DoExperiment1or2()
	{
		LinkedHashMap<String, Batch_Ins> CreatedDataset = new LinkedHashMap<String, Batch_Ins>();
		
		LinkedHashMap<String, Batch_Ins> LowBatchInstances = new LinkedHashMap<String, Batch_Ins>();
		LinkedHashMap<String, Batch_Ins> MediumBatchInstances = new LinkedHashMap<String, Batch_Ins>();
		LinkedHashMap<String, Batch_Ins> HighBatchInstances = new LinkedHashMap<String, Batch_Ins>();
		
		Read_WholeDataset("StefanFiles_HCOMP/datasets/Batch_2984090_batch_results_low_8000.csv", LowBatchInstances);
		Read_WholeDataset("StefanFiles_HCOMP/datasets/Batch_2984078_batch_results_medium_4000.csv", MediumBatchInstances);
		Read_WholeDataset("StefanFiles_HCOMP/datasets/Batch_2984071_batch_results_high_4000.csv",HighBatchInstances);
		
		Compute_MajorityLabels(LowBatchInstances, 8);
		Compute_MajorityLabels(MediumBatchInstances, 4);
		Compute_MajorityLabels(HighBatchInstances, 4);
		
		LinkedHashMap<String, Batch_Ins> MixedDataset = new LinkedHashMap<String, Batch_Ins>();
		MixedDataset.putAll(LowBatchInstances);
		MixedDataset.putAll(MediumBatchInstances);
		MixedDataset.putAll(HighBatchInstances);
		
		Generate_LATweets_List(MixedDataset);
		ReadLabelledStefanFile("train_twitter.csv");
		
		MergeTwoMaps();
		
		int datasetSize = 1100;
		int NSplits = 3;
				
		randomGenerator.setSeed(13);
		
		double per = 0.0;
		int sizeGet = 0;
		for (int i = 0; i <= 100; i++) {
			CreatedDataset = new LinkedHashMap<String, Batch_Ins>();
			per = i/100.0;
			sizeGet = (int)Math.floor(per*datasetSize);
			for(int j=0; j<NSplits; j++)
			{
				try {
					CreatedDataset = Create_DatasetWith_GivenLA_TweetPercentage(datasetSize ,sizeGet);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}	
				WriteOut_Encoding("LA_Tweets_Percentage_" + datasetSize + "/LowTweetSize_" + sizeGet + "_datasetSize_" + datasetSize + "_sample_" + j + ".txt", CreatedDataset);
			}
		}
	}
	
	//For both of these, I added 1 tweet for each iteration till the maximum.
	//Experiment 3: Dataset size is 174 and the maximum is the 50% of the dataset is LA tweets.
	//Experiment 4: Dataset size is 87 and the maximum is the 100% of the dataset is LA tweets.
	public static void DoExperiment3or4()
	{
		LinkedHashMap<String, Batch_Ins> CreatedDataset = new LinkedHashMap<String, Batch_Ins>();
		
		LA_DatasetMerged = ReadSecondType_DataFiles("twitter_agreement_low_low_agree_174_8_labels.csv");
		NotLA_DatasetMerged = ReadSecondType_DataFiles("twitter_agreement_low_high_agree_174_8_labels.csv");
		
		int datasetSize = 87;
		int LA_Size = 87;
		int NSplits = 3;
				
		randomGenerator.setSeed(13);
		
		for (int sizeGet = 0; sizeGet <= LA_Size; sizeGet++) {
			CreatedDataset = new LinkedHashMap<String, Batch_Ins>();
			//per = i/100.0;
			//sizeGet = (int)Math.floor(per*datasetSize);
			for(int j=0; j<NSplits; j++)
			{
				try {
					CreatedDataset = Create_DatasetWith_GivenLA_TweetPercentage(datasetSize ,sizeGet);
				} catch (IOException e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}	
				WriteOut_Encoding("LA_Tweets_Percentage_" + datasetSize + "/LowTweetSize_" + sizeGet + "_datasetSize_" + datasetSize + "_sample_" + j + ".txt", CreatedDataset);
			}
		}
	}
	//In order to read experiment3 and 4 data files
	public static LinkedHashMap<Integer, Batch_Ins> ReadSecondType_DataFiles(String filename)
	{
		LinkedHashMap<Integer, Batch_Ins> Map = new LinkedHashMap<Integer, Batch_Ins>();
		
		Batch_Ins BI = new Batch_Ins();
		String ID = "", tweet = "", ans = "";
		try {		

			File f = new File(filename);
			BufferedReader in = new BufferedReader(
			   new InputStreamReader(
	                      new FileInputStream(f), "UTF8"));

			//BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";

			//in.readLine();
			int keyVal = 0;
			while ((line = in.readLine()) != null) 
			{
				BI = new Batch_Ins();
				
				String [] obj = line.split("\t");
				ID = obj[0];
				tweet = obj[1];
				
				BI.ID = ID;
				BI.tweet = tweet;
				//BI.labels.add(ans);				
				Map.put(keyVal, BI);	
				
				keyVal++;	
			}
			in.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
		return Map;
	}
	public static void WriteOut_Encoding(String outFile, LinkedHashMap<String, Batch_Ins> FinalDataset)
	{
		try {
			Writer out = new BufferedWriter(new OutputStreamWriter(
				    new FileOutputStream(outFile), "UTF-8"));
			
			Batch_Ins BI = new Batch_Ins();
			for (Entry<String, Batch_Ins> ent : FinalDataset.entrySet()) {
				BI = new Batch_Ins();
				BI = ent.getValue();
				
				out.write(BI.ID + "\t" + BI.tweet);
				out.write("\n");
			}
			out.close();
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	
	public static LinkedHashMap<String, Batch_Ins> Create_DatasetWith_GivenLA_TweetPercentage(int datasetSize, int LASize_InDataset) throws IOException
	{
		//System.out.println("Running");
		/*Writer out = new BufferedWriter(new OutputStreamWriter(
			    new FileOutputStream(LASize_InDataset + ".txt")));*/
		
		LinkedHashMap<String, Batch_Ins> FinalDataset = new LinkedHashMap<String, Batch_Ins>();
		List<Integer> IndicesLow = new ArrayList<Integer>();
		
		int LASize = LA_DatasetMerged.size();
		//Random randomGenerator = new Random();	
		while(IndicesLow.size() < LASize_InDataset)
		{
			int randomInt = randomGenerator.nextInt(LASize);
			//System.out.println(randomInt);
			if(!IndicesLow.contains(randomInt))
				IndicesLow.add(randomInt);
		}
		
		
		List<Integer> IndicesRest = new ArrayList<Integer>();
		int RestSize = datasetSize - LASize_InDataset;
		//randomGenerator = new Random();
		
		int restTweetSize = NotLA_DatasetMerged.size();
		while(IndicesRest.size() < RestSize)
		{
			int randomInt = randomGenerator.nextInt(restTweetSize);
			//System.out.println(randomInt);
			if(!IndicesRest.contains(randomInt))
				IndicesRest.add(randomInt);
		}
		
		Batch_Ins BI = new Batch_Ins();
		for (Integer index : IndicesLow) {
			BI = new Batch_Ins();
			BI = LA_DatasetMerged.get(index);
			FinalDataset.put(BI.ID, BI);		
		}
		
		for (Integer index1 : IndicesRest) {
			BI = new Batch_Ins();
			BI = NotLA_DatasetMerged.get(index1);
			FinalDataset.put(BI.ID, BI);
		}
		
		if(IndicesLow.size() + IndicesRest.size() != datasetSize)
			System.out.println("FALSE1");
		if(IndicesLow.size() != LASize_InDataset)
			System.out.println("FALSE2");
		
		//out.close();
		return FinalDataset;
		
	}
	
	public static void MergeTwoMaps()
	{
		int lowInd = 0;
		int medHighInd = 0;
		
		for (Entry<Integer, Batch_Ins> ent : LA_Dataset.entrySet()) {
			LA_DatasetMerged.put(lowInd, ent.getValue());
			lowInd++;
		}
		for (Entry<Integer, Batch_Ins> ent : LA_DatasetStefan.entrySet()) {
			LA_DatasetMerged.put(lowInd, ent.getValue());
			lowInd++;
		}
		
		
		for (Entry<Integer, Batch_Ins> ent : NotLA_Dataset.entrySet()) {
			NotLA_DatasetMerged.put(medHighInd, ent.getValue());
			medHighInd++;
		}
		for (Entry<Integer, Batch_Ins> ent : NotLA_DatasetStefan.entrySet()) {
			NotLA_DatasetMerged.put(medHighInd, ent.getValue());
			medHighInd++;
		}
		
		System.out.println("LA Tweets Size: " + LA_DatasetMerged.size());
		System.out.println("Not-LA Tweets Size: " + NotLA_DatasetMerged.size());
	}
	public static void ReadLabelledStefanFile(String filename)
	{
		int lowBegInd = 0;
		int medHighBegInd = 0;
		
		Batch_Ins BI = new Batch_Ins();

		int count = 0;
		try {		

			File f = new File(filename);
			BufferedReader in = new BufferedReader(
			   new InputStreamReader(
	                      new FileInputStream(f), "UTF8"));

			//BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";

			//in.readLine();
			String majLabel = "";
			while ((line = in.readLine()) != null) 
			{
				count++;
				String [] obj = line.split("\t");
				BI = new Batch_Ins();
				BI.ID = obj[0];
				if(obj[1].equals("Negative"))
					majLabel = "Neg";
				else if(obj[1].equals("Positive"))
					majLabel = "Pos";
				else if(obj[1].equals("Neutral"))
					majLabel = "Neut";
				else if(obj[1].equals("Irrelevant"))
					majLabel = "Irrel";
				
				BI.majorityLabel = majLabel;
				BI.tweet = obj[3];
				
				if(obj[2].equals("low"))
				{
					LA_DatasetStefan.put(lowBegInd, BI);
					lowBegInd++;
				}
				else if(obj[2].equals("medium") || obj[2].equals("high"))
				{
					NotLA_DatasetStefan.put(medHighBegInd, BI);
					medHighBegInd++;
				}
			}
			//System.out.println(count);
			in.close();
		}catch(Exception e)
		{
			e.printStackTrace();
		}
	}
	
	public static void Generate_LATweets_List(LinkedHashMap<String, Batch_Ins> MixedDataset)
	{
		int newInd = 0;
		for (Entry<String, Batch_Ins> ins : MixedDataset.entrySet()) 
		{
			Batch_Ins BI = new Batch_Ins();
			BI = ins.getValue();
			if(BI.LA == 1)
			{
				LA_Dataset.put(newInd, BI);
			}
			else
			{
				NotLA_Dataset.put(newInd, BI);
			}
			newInd++;
		}
	}
	
	public static void FindLA_Tweets(LinkedHashMap<String, Batch_Ins> Map)
	{
		LinkedHashMap<String, Integer> Values = new LinkedHashMap<String, Integer>();
		
		Batch_Ins BI = new Batch_Ins();
		for (Entry<String, Batch_Ins> ent : Map.entrySet()) {
			BI = new Batch_Ins();
			BI = ent.getValue();
		}
	}
	
	public static void Compute_MajorityLabels(LinkedHashMap<String, Batch_Ins> Map, int labelCount)
	{
	
		LinkedHashMap<String, Integer> Values = new LinkedHashMap<String, Integer>();
		int numOfAgreed = 0;
		
		Batch_Ins BI = new Batch_Ins();
		int Pos = 0, Neg = 0, Neut = 0, Irrel = 0;
		for (Entry<String, Batch_Ins> ent : Map.entrySet()) 
		{
			Values = new LinkedHashMap<String, Integer>();
			
			Pos = 0; 
			Neg = 0; 
			Neut = 0;
			Irrel = 0;
			
			BI = new Batch_Ins();
			BI = ent.getValue();
			
			for (String label : BI.labels) 
			{
				if(label.equals("Highly Relevant"))
					Neg++;
				else if(label.equals("Relevant"))
					Neut++;
				else if(label.equals("Not Relevant"))
					Pos++;
				else if(label.equals("I can't judge"))
					Irrel++;
			}
			Values.put("Pos", Pos);
			Values.put("Neut", Neut);
			Values.put("Neg", Neg);
			Values.put("Irrel", Irrel);
					
			ent.getValue().numOfPositives = Pos;
			ent.getValue().numOfNegatives = Neg;
			ent.getValue().numOfNeutrals = Neut;
			ent.getValue().numofIrrelevants = Irrel;
			
			int maxVal = Collections.max(Values.values());
			String majLabel = "";
			for (Entry<String, Integer> elt : Values.entrySet()) 
			{
				if(elt.getValue() == maxVal)
				{
					majLabel = elt.getKey();
					numOfAgreed = elt.getValue();
					ent.getValue().majorityLabel = majLabel;
					ent.getValue().numOfAnnotatorsAgreedOnMajLabel = numOfAgreed;
					if(numOfAgreed <= labelCount/2)
						ent.getValue().LA = 1;
					
					break;
				}
			}
		}
	}
	
	public static void Read_WholeDataset(String filename, LinkedHashMap<String, Batch_Ins> Map)
	{
		Batch_Ins BI = new Batch_Ins();
		String ID = "", tweet = "", ans = "";
		int label = 0;
		try {		

			File f = new File(filename);
			BufferedReader in = new BufferedReader(
			   new InputStreamReader(
	                      new FileInputStream(f), "UTF8"));

			//BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";

			in.readLine();
			while ((line = in.readLine()) != null) 
			{
				BI = new Batch_Ins();
				
				String [] obj = line.split("\",\"");
				ID = obj[27];
				tweet = obj[29];
				ans = obj[30].replace("\"", "");
				//System.out.println(tweet);
	
/*				if(ans.equals("\"Highly Relevant\""))
					label = 0;
				else if(ans.equals("\"Relevant\""))
					label = 1;
				else if(ans.equals("\"Not Relevant\""))
					label = 2;
				else if(ans.equals("\"I can't judge\""))
					label = -1;*/
				
				if(!Map.containsKey(ID))
				{
					BI.ID = ID;
					BI.tweet = tweet;
					BI.labels.add(ans);				
					
					Map.put(ID, BI);				
				}
				else
				{
					BI = Map.get(ID);
					BI.labels.add(ans);			
				}				
			}
			in.close();
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
				
				
	}
	
	public static void ProcessEncoding_Dataset(String filename)
	{
		Writer out;
		//PrintWriter pw = new PrintWriter("data.csv");
		Pattern p = Pattern.compile("[\uD800-\uDFFF].", Pattern.UNICODE_CHARACTER_CLASS);
		p = Pattern.compile("[^\u0000-\uFFFF]", Pattern.UNICODE_CHARACTER_CLASS);
		try {
			out = new BufferedWriter(new OutputStreamWriter(
				    new FileOutputStream(filename), "UTF-8"));
			
			File f = new File("high.csv");
			

			BufferedReader in = new BufferedReader(
			   new InputStreamReader(
	                      new FileInputStream(f), "UTF8"));

			//BufferedReader b = new BufferedReader(new FileReader(f));
			String line = "";

			in.readLine();
			out.write("tweetID,query,tweet\n");
			while ((line = in.readLine()) != null) {
				String [] obj = line.split(",");
				
				Matcher matcher = p.matcher(line);
				if (matcher.find())
				{
					//System.out.println(line);
					line = line.replaceAll(p.toString(), "");
					//System.out.println(line);
				}
				out.write(line);
				out.write("\n");
			}
			out.close();
			in.close();
		} catch (UnsupportedEncodingException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (FileNotFoundException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
