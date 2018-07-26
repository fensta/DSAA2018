import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;
import java.util.Arrays;
import java.util.List;


public class CheckFiles {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stubtry {		

		String folderName = "FeatureFiles_Second/";
		File folderFeatures = new File(folderName);
		List<File> listOfFilesFeatures = Arrays.asList(folderFeatures.listFiles());
		

		for (int i = 0; i < listOfFilesFeatures.size(); i++) 
		{
			if (listOfFilesFeatures.get(i).isFile()) 
			{		
				String filename = listOfFilesFeatures.get(i).getName();
				File f = new File("FeatureFiles_Second/" + filename);
				BufferedReader in = new BufferedReader(
				   new InputStreamReader(
		                      new FileInputStream(f), "UTF8"));

				//BufferedReader b = new BufferedReader(new FileReader(f));
				String line = "";

				//in.readLine();
				while ((line = in.readLine()) != null) 
				{
					String [] obj = line.split("\t");
					String [] obj1 = obj[1].split(",");
					if(obj1.length != 94)
						System.out.println(filename + ":" + obj1.length);
				}
				in.close();
			}
		}
		
		

	}

}
