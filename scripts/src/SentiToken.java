import java.util.ArrayList;
import java.util.List;


public class SentiToken {
	
	String ID;
	String POSTag;
	double posPol;
	double negPol;
	
	double avgPol;
	double domPol;
	
	List<Double> posScoreList = new ArrayList<Double>(); //synset
	List<Double> negScoreList = new ArrayList<Double>(); //synset
	String term;
}
