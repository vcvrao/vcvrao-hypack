
/*******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Code Name      : RankCalculation.java

 Objective      : Write a java concurrent APIs program for Rank calculation
                  for a large Array

 Input          : Double Array Length
	          Number of Threads

 Output         : Rank of each array element with execution time for
                  serial/parallel/Executor API

 APIs Used      : Java Concurrent APIs - Executor API

 Created        : August-2013

 E-mail         : hpcfte@cdac.in     

*****************************************************************************/

package concurrent;

import java.util.concurrent.*;
/*The rank of the element in the array is the number of smaller elements in the array 
plus number of equal elements that appear to its left*/

/* Thread class for create a thread */
class RankThread extends Thread{

	/* Instance Variable declaration */
	int [] pointToCommonArray;
	double [] seqArray;
	int threadNo;
	int initialValue;
	int eachPartValue,threadArrayStart,threadArrayEnd;

	/* Thread class constructor takes thread no., 
	no.of array elements to which ranks calculated by this thread,
	rank is calculated for this array elements,common array which stored rank values*/
	public RankThread(int threadNo,int eachPartValue,double [] seqArray,int [] commonArray){
		pointToCommonArray = commonArray;
		this.seqArray=seqArray;
		this.threadNo = threadNo;
		this.eachPartValue =eachPartValue;
		/*calculate loop start and end point for this thread */
		threadArrayStart =(threadNo*eachPartValue) ;
		threadArrayEnd   = ((threadNo+1)*eachPartValue-1); 		
	}

	/* Thread class overriden run() method */
	public void run (){

		int counter=0;
		/*calculate ranks for seqArray elements from threadArrayStart to 
		threadArrayEnd and put result as a pointToCommonArray element*/
		for(counter=threadArrayStart; counter<=threadArrayEnd; counter++) { 
			/*calculate ranks for seqArray elements by calling calculateRank() method */
			pointToCommonArray[counter]=RankCalculation.calculateRank(seqArray[counter],seqArray,counter);
		}

	}
}

/* Main class */
public class RankCalculation {

	/*static method which takes no. to which rank is calculated,
	array to which the no. is a element,index of the no. in the array,
	return rank of the no.*/	
	public static int calculateRank(double no,double [] dArray, int position){
		
		/*intialize rank variable to default value 0*/
		int rank = 0;
		/*check smaller no. than rank calculated no. 
		in the array if there then increment rank by 1*/
		for(int i = 0; i < dArray.length; i++ ){
			if(no > dArray[i]){
				rank++;
			}
		}
		/*check the no. which are equal to rank calculated no. 
		that appear to its left in the array if there then increment rank by 1*/
		for(int i = position-1; i >= 0; i-- ){
			if(no == dArray[i])
				rank++;	
		}
		return rank;
		
	}
		
	/* Main() method which takes two commandline arguments Array Size 
	and number of threads */
	public static void main(String[] args) throws Exception{

		/* check the number of commandline argument */
		if(args.length !=2){
			System.out.println("Invalid Arguments <Array Size> <Number of threads>");
			return;
		}
		/* Takes the  Array size and number of threads given by commandline arguments*/
		int dArraySize = Integer.parseInt(args[0].trim());
		int numberOfThreads = Integer.parseInt(args[1].trim());
		
		/* Calculate the Array length for each thread */
		int eachPartValue = dArraySize/numberOfThreads;
		Thread [] threads = new RankThread[numberOfThreads];

		long startTime = 0;
		long endTime = 0;
		double [] dArray = new double[dArraySize];
		/*fill the array elements in decending order */
		for(int i = 0; i < dArraySize; i++){
			dArray[i]=(dArraySize-i)*1.0;
			
		}

		/* Create the intermediate arrays which is used for rank calculation */
		int []pArray = new int [dArray.length];
		int []fArray = new int [dArray.length];

		/*Takes the start time for serial rank calculation */
		startTime = System.currentTimeMillis();
		/*calculate the rank for all array elements*/
		for(int i = 0; i < dArray.length; i++){
			pArray[i]=calculateRank(dArray[i],dArray,i);
		}
		/*Takes the end time for serial rank calculation */
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in serial Rank Calculation is :"+ (endTime-startTime)/(1000.0)+" Seconds");
 
		/*Takes the start time for parallel rank calculation */
		startTime = System.currentTimeMillis();
		/*create the threads with thread no.,length of array which is used by particular thread,
		array to which elements rank is calculated,common result array where rank is stored*/
		for(int i = 0; i < numberOfThreads; i++ ){
			threads[i] = new RankThread(i,eachPartValue,dArray,fArray);
			threads[i].start();
		}
		/* main thread waits for all other threads */	
		for(int i = 0; i < numberOfThreads; i++ )
			threads[i].join();
		/*Takes the end time for parallel rank calculation */
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in parallel Rank Calculation is :"+ (endTime-startTime)/(1000.0)+" Seconds");

		/*Takes the start time for parallel rank calculation using Executor (Java Concurrent API)*/ 
		startTime = System.currentTimeMillis();
		/*Create a thread pool of no. of threads given by user */
                ExecutorService e = Executors.newFixedThreadPool(numberOfThreads);
		/*Create a CountDownLatch with no of threads it has to wait */
                final CountDownLatch doneSignal = new CountDownLatch(numberOfThreads);

		/*create a method innner class for create a thread*/
		class RankThread extends Thread{

			/* Instance Variable declaration */
        		int [] pointToCommonArray ;
        		double [] seqArray;
        		int threadNo ;
        		int eachPartValue,threadArrayStart,threadArrayEnd;

			/* Thread class constructor takes thread no., 
			no.of array elements to which ranks calculated by this thread,
			rank is calculated for this array elements,common array which stored rank values*/
       			public RankThread(int threadNo,int eachPartValue,double [] seqArray,int [] commonArray){
                		pointToCommonArray = commonArray;
                		this.seqArray=seqArray;
                		this.threadNo = threadNo;
                		this.eachPartValue =eachPartValue;
                		threadArrayStart =(threadNo*eachPartValue) ;
                		threadArrayEnd   = ((threadNo+1)*eachPartValue-1);
        		}

			/* Thread class overriden run() method */
        		public void run (){
                		int  counter;

			/*calculate ranks for seqArray elements from threadArrayStart 
			to threadArrayEnd and put result as a pointToCommonArray element*/
                		for(counter=threadArrayStart; counter<=threadArrayEnd; counter++) { 
                        		pointToCommonArray[counter]=RankCalculation.calculateRank(seqArray[counter],seqArray,counter);
                		}
				/*decrease the countDownLatch value by 1 */
				doneSignal.countDown();
        		}
		}

		/* assign the work to pool threads by create a new thread object using thread no., 
		no.of array elements to which ranks calculated by this thread,
		array for which elements rank is calculated ,common array which stored rank values*/
                for(int i = 0; i < numberOfThreads; i++ )
                        e.execute(new RankThread(i,eachPartValue,dArray,fArray));
                try {
                         // wait for all to finish
                        doneSignal.await(); 
                } catch (InterruptedException ie) {
                }
		/*shut down the Executor service */
                e.shutdown();
	
		/*Takes the start time for parallel rank calculation using Executor (Java Concurrent API)*/ 
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in parallel concurrent Rank Calculation is :"+ (endTime-startTime)/(1000.0)+" Seconds");
	}
}
