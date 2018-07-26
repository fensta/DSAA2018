import java.util.ArrayList;
import java.util.List;


public class Batch_Ins {
	
	String ID;
	String tweet;
	List<String> labels = new ArrayList<String>();
	String majorityLabel;
	
	int LA = 0;
	int numOfAnnotatorsAgreedOnMajLabel;
	
	int numOfNegatives = 0;
	int numOfPositives = 0;
	int numOfNeutrals = 0;
	int numofIrrelevants = 0;

}
