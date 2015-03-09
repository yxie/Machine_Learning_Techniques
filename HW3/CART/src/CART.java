import java.util.*;
import java.lang.Math;
public class CART {
	//Original DATA
	private List<Double> x1 = new ArrayList<Double>();
	private List<Double> x2 = new ArrayList<Double>();
	private List<Integer> y = new ArrayList<Integer>();
	//Decision Tree
	/*
	private List<Integer> nid = new ArrayList<Integer>();
	private List<Integer> lnode = new ArrayList<Integer>();
	private List<Integer> rnode = new ArrayList<Integer>();
	private List<Integer> feature_list = new ArrayList<Integer>();
	private List<Double> theta_list = new ArrayList<Double>();
	private List<Boolean> is_leaf = new ArrayList<Boolean>();
	private int id = 0;
	*/
	private Node root;
	private int id_count=0;
	//Solutions
	private int internal=0;
	
	public CART(String filename){                   
		if(filename == null)
			throw new NullPointerException("null file");
		In in = new In(filename);
		//100 train data
		String line;
		String[] words;
		for(int i=0; i<100; i++){
			line = in.readLine();
			words = line.split(" ");
			if(words.length!=3){
				throw new NullPointerException("read data file error");
			}
			x1.add(Double.parseDouble(words[0]));
			x2.add(Double.parseDouble(words[1]));
			y.add(Integer.parseInt(words[2]));
		}
	}
	
	private void DT(List<Double> x1, List<Double> x2, List<Integer>y, Node node){
		int N = x1.size();
		double impurity_opt = Double.MAX_VALUE;
		int feature_opt = 1;
		double theta_opt = 0;
		
		
		
		if(impurity(x1, y, Double.MIN_VALUE) == 0 || sameX(x1, x2)){
			 node.g = y.get(0);
			 node.is_leaf = true;
			 StdOut.println(node.id + " is leaf");
			 return;
		}
		node.is_leaf = false;
		node.g = 0;
		internal++;

		double[] x1_t = new double[N];
		double[] x2_t = new double[N];
		for(int i=0; i<N; i++){
			x1_t[i] = x1.get(i);
			x2_t[i] = x2.get(i);
		}
		//x1
		Arrays.sort(x1_t);
		for(int n=0; n<N-1; n++){
			double theta = (x1_t[n] + x1_t[n+1]) / 2;
			//StdOut.println(theta);
			double imp   = impurity(x1, y, theta);
			if(imp < impurity_opt){
				impurity_opt = imp;
				feature_opt = 1; //x1
				theta_opt = theta;
			}
		}
		//x2
		Arrays.sort(x2_t);
		for(int n=0; n<N-1; n++){
			double theta = (x2_t[n] + x2_t[n+1]) / 2;
			double imp   = impurity(x1, y, theta);
			if(imp < impurity_opt){
				impurity_opt = imp;
				feature_opt = 2; //x1
				theta_opt = theta;
			}
		}
		
		StdOut.println(node.id + " " + N + " " + feature_opt + " " + theta_opt + " " + impurity_opt);
		//new java.util.Scanner(System.in).nextLine();
		
		node.feature = feature_opt;
		node.theta = theta_opt;
		
		List<Double> x1_p = new ArrayList<Double>();
		List<Double> x2_p = new ArrayList<Double>();
		List<Integer> y_p = new ArrayList<Integer>();
		List<Double> x1_n = new ArrayList<Double>();
		List<Double> x2_n = new ArrayList<Double>();
		List<Integer> y_n = new ArrayList<Integer>();
		
		if(feature_opt == 1){
			for(int i=0; i<N; i++){
				 if(x1.get(i) >= theta_opt){
					x1_p.add(x1.get(i));
					x2_p.add(x2.get(i));
					y_p.add(y.get(i));
				}
				else{
					x1_n.add(x1.get(i));
					x2_n.add(x2.get(i));
					y_n.add(y.get(i));
				}
			}
		}
		else if(feature_opt == 2){
			for(int i=0; i<N; i++){
				 if(x2.get(i) >= theta_opt){
					x1_p.add(x1.get(i));
					x2_p.add(x2.get(i));
					y_p.add(y.get(i));
				}
				else{
					x1_n.add(x1.get(i));
					x2_n.add(x2.get(i));
					y_n.add(y.get(i));
				}
			}
		}
		//right tree (>theta)
		id_count++;
		Node right = new Node(id_count);
		node.rnode = right;
		DT(x1_p, x2_p, y_p, right);
		//left tree (<theta)
		id_count++;
		Node left = new Node(id_count);
		node.lnode = left;
		DT(x1_n, x2_n, y_n, left);
		return;
	}
	
	private boolean sameX(List<Double> x1, List<Double> x2){
		double a = x1.get(0);
		double b = x2.get(0);
		int size = x1.size();
		boolean result = true;
		for(int i=1; i<size; i++){
			if(x1.get(i) != a || x2.get(i) != b){
				result = false;
				break;
			}
		}
		return result;
	}
	
	private double impurity(List<Double> x, List<Integer> y, double theta){
		double result=0;
		double count1_p=0;
		double count1_n=0;
		double count2_p=0;
		double count2_n=0;
		int N = x.size();
		for(int i=0; i<N; i++){
			if(x.get(i) >= theta && y.get(i) == 1)
				count1_p++;
			else if(x.get(i) >= theta && y.get(i) == -1)
				count1_n++;
			else if(x.get(i) < theta && y.get(i) == 1)
				count2_p++;
			else if(x.get(i) < theta && y.get(i) == -1)
				count2_n++;
		}
		if(theta == Double.MIN_VALUE ){
			if (count1_n==0 || count1_p==0){
				StdOut.println("find impurity = 0");
				return 0;
			}
			else{
				return 1;
			}
		}
		double count1 = count1_p + count1_n;
		double count2 = count2_p + count2_n;
		double Gini1 = 1 - Math.max((count1_p)/count1, (count1_n)/count1);
		double Gini2 = 1 - Math.max((count2_p)/count2, (count2_n)/count2);
		result = count1 * Gini1 + count2 * Gini2;
		//StdOut.println(theta + " " + count1_p + " " + count1_n + " " + count2_p + " " + count2_n + " " + Gini1 + " " + Gini2 + " " + result);
		//new java.util.Scanner(System.in).nextLine();
		
		return result;
	}
	
	
	private int prediction(double x1, double x2, Node n){
		int result=0;
		if(n.is_leaf){
			result =  n.g;
		}
		else{
			if(n.feature == 1){
				if(x1 >= n.theta)
					result = prediction(x1, x2, n.rnode);
				else
					result = prediction(x1, x2, n.lnode);
			}
			else if(n.feature == 2){
				if(x2 >= n.theta)
					result = prediction(x1, x2, n.rnode);
				else
					result = prediction(x1, x2, n.lnode);
			}
		}
		return result;
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		/* test case
		CART c = new CART("test.dat");
		c.root = new Node(0);
		c.DT(c.x1, c.x2, c.y, c.root);
		int p = c.prediction(0, 0, c.root);
		StdOut.println(p);
		p = c.prediction(1.2, 1.2, c.root);
		StdOut.println(p);
		p = c.prediction(1.8, 1.8, c.root);
		StdOut.println(p);
		p = c.prediction(2.2, 2.2, c.root);
		StdOut.println(p);
		p = c.prediction(2.8, 2.8, c.root);
		StdOut.println(p);
		*/
		CART c = new CART("hw3_train.dat");
		c.root = new Node(0);
		c.DT(c.x1, c.x2, c.y, c.root);
		StdOut.println(c.internal);
	}

}

class Node{
	int id;
	Node lnode;
	Node rnode;
	double theta;
	int feature;
	boolean is_leaf = false;
	int g;
	
	Node(int id){
		this.id = id;
	}
}
