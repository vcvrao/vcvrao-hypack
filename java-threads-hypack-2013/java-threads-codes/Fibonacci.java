/*************************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                   October 15-18, 2013

 Code Name           : Fibonacci.java

 Objective           : Write a java concurrent APIs for calculate Fibonacci series
                               
 Input               :  Number that we want to calculate Fibonacci number
                                  
 Output              :  Fibonacci number with execution time for serial/parallel/Executor API

 APIs Used           : Java Concurrent APIs -Executor API

 Created             : August-2013

 E-mail              : hpcfte@cdac.in     

******************************************************************************/

package concurrent;

import java.util.concurrent.*;

/*Thread class for create thread */
class FiboThread extends Thread{

	/*Instance variable declaration */
	double number; 
	int threadNo;
	Double []results;

	/*constructor for thread class */
	public FiboThread(int threadNo,double number,Double [] results){
		this.number = number;
		this.threadNo = threadNo;
		this.results = results;
	}

	/*Thread's overriden run() method 
	which calculate fibonacci no */
	public void run (){
		if(number==0.0 || number==1.0)
			results[threadNo] = number;
		else
			results[threadNo] = Fibonacci.calculateFiboNo(number-1)+Fibonacci.calculateFiboNo(number-2);

	}
}

/*Main class */
public class Fibonacci {

	/*static recursive method for calculate fibonacci no. */
	public static double calculateFiboNo(double no){
		/*if no. is 0 or 1 return no. otherwise recursively 
		call calculateFiboNo() for (no.-1 + no-2)*/
		if(no ==0.0 || no==1.0)
			return no;
		else if(no < 0.0)
			return 0.0;
		else
			return calculateFiboNo(no-1.0)+calculateFiboNo(no-2.0);
	}

	/* main() method takes one argument no. which we want 
	to calculate fibonacci no.*/		
	public static void main(String[] args) throws Exception{
	
		if(args.length !=1){
			System.out.println("Invalid Arguments <number>");
			return;
		}
		int intNumber = Integer.parseInt(args[0].trim());
		double number = intNumber*1.0;
		int numberOfThreads = 2;
		
		Thread [] threads = new FiboThread[numberOfThreads];

		/*Initialize the local variables */
		long startTime = 0;
		long endTime = 0;

		/*Takes the start time for serial fibinacci no. calculation */
		startTime = System.currentTimeMillis();
		System.out.println("Fibonacci No. is "+calculateFiboNo(number));

		/*Takes the start time for serial fibinacci no. calculation */
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in serial Fibonacci no Calculation is :"+ (endTime-startTime)/(1000.0)+" Seconds");
 
		Double [] results = new Double[2]; 

		/*Takes the start time for parallel fibinacci no. calculation */
		startTime = System.currentTimeMillis();

		/*create a two threads by FiboThread object using thread no.,
		no. which we want to calculate fibonacci no.,result array*/
		for(int i = 0; i < numberOfThreads; i++ ){
			double localno = number-i-1.0;
			if (localno==0.0 && i==0)
				localno = 1.0;
			
			threads[i] = new FiboThread(i,localno,results);
			threads[i].start();
		}

		/*main thread waits for other threads to finish */
		for(int i = 0; i < numberOfThreads; i++ )
			threads[i].join();

		/*print the fibonacci no. of given no.*/
    		System.out.println("Fibonacci No. "+(results[0]+results[1]));	

		/*Takes the end time for parallel fibinacci no. calculation */
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in parallel Fibonacci no Calculation is :"+ (endTime-startTime)/(1000.0)+" Seconds");
                
		/*Takes the start time for parallel fibinacci no. 
		calculation with Executor Service*/
		startTime = System.currentTimeMillis();
                ExecutorService e = Executors.newFixedThreadPool(numberOfThreads);
                final CountDownLatch doneSignal = new CountDownLatch(numberOfThreads);

		/*create method inner class for create FiboThread*/
		class FiboThread extends Thread{

			/*define instance variables*/
        		double number;
        		int threadNo;
        		Double []results;

			/*inner class constructor */
        		public FiboThread(int threadNo,double number,Double [] results){
                		this.number = number;
                		this.threadNo = threadNo;
                		this.results = results;
        		}

			/* inner thread class overriden run method */
        		public void run (){
				if(number==0.0 || number==1.0)
					results[threadNo] = number;
				else
                			results[threadNo] = Fibonacci.calculateFiboNo(number-1)+Fibonacci.calculateFiboNo(number-2);
				doneSignal.countDown();
        		}
		}
		Double [] resultsC = new Double[2]; 

		/*create the two threads for fibonacci no. claculation*/
                for(int i = 0; i < numberOfThreads; i++ ){
			double localno = number-i-1.0;
			if (localno==0.0 && i==0)
				localno = 1.0;
                        e.execute(new FiboThread(i,localno,resultsC));
		}
                try {
                        doneSignal.await(); // wait for all to finish
                } catch (InterruptedException ie) {
                }

		/* shutdown the Executor service pool*/
                e.shutdown();
    		System.out.println("Fibonacci No. "+(resultsC[0]+resultsC[1]));	

		/*Takes the start time for parallel fibinacci no. calculation with Executor Service*/
                endTime = System.currentTimeMillis();
                System.out.println("Total elapsed time in parallel concurrent Fibonacci Calculation is :"+ (endTime-startTime)/(1000.0)+" Seconds");

	}
}
