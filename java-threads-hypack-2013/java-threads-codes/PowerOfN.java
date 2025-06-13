/******************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Code Name          : PowerOfN.java

 Objective          : Write a java concurrent APIs program for calculate power of 1.0 

 Input              : Power 
	              Number of Threads

 Output             : Power calculation of 1.0 with execution time for 
		      serial/parallel/Executor API

 APIs Used          : Java Concurrent APIs -Executor API


 Created            : August-2013

 E-mail             : hpcfte@cdac.in     

**********************************************************************************/

package concurrent;

/*import java concurrent package*/
import java.util.concurrent.*;

/*Thread class for power calculation*/
class PowerThread extends Thread{

	double number; 
	long power;
	int threadNo;
	Double []results;

	/*Thread class constructor takes thread no.,number which we want to calculate power,
	power(how much power value),common results array*/
	public PowerThread(int threadNo,double number,long power,Double [] results){
		this.number = number;
		this.power = power;
		this.threadNo = threadNo;
		this.results = results;
	}

	/*Thread class overriden run() method*/
	public void run (){
		System.out.println(threadNo +" Thread Start "+number+" End "+power);
		/*call static calculatePower() method to calculate given power of given no.*/
		results[threadNo] = PowerOfN.calculatePower(number,power);

	}
}

/*Main class*/
public class PowerOfN {

	/*static method to calculate given power of given double no.*/	
	public static double calculatePower(double no,long power){
		
		/*initialze result variable with default value 1.0 so we can used for multiply
		  with no.*/
		double result =1.0;
		/*To get power of no. multiplies the no. by given power value times*/
		for(int i = 0; i < power; i++)
			result*=no;
		return result;	
	}
			
		
	/*main() method takes power value and number of thread as a commandline argument*/	
	public static void main(String[] args) throws Exception{

		/*checks commandline argument validity*/
		if(args.length!=2){
			System.out.println("Invalid Arguments <power> <number of threads>");
			return;
		}

		/*Takes power value to be calculated from commandline argument*/
		long power = Long.parseLong(args[0].trim());
		double number = 1.0;
		int numberOfThreads = Integer.parseInt(args[1].trim());
		
		long startTime = 0;
		long endTime = 0;
		
		/*Takes start time for serial power calculation*/
		startTime = System.currentTimeMillis();
		System.out.println(number+" to the Power of "+power +" is "+calculatePower(number,power));
		/*Takes end time for serial power calculation*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in serial power Calculation is :"+ (endTime-startTime)/(1000.0)+" Seconds");
 
		/*calculate power value for ecah thread from commandline argument*/
		long eachPartValue = power/numberOfThreads;
		System.out.println("each part value "+eachPartValue);
		
		Thread [] threads = new PowerThread[numberOfThreads];
		/*results array used by each thread put it's result as a array element*/
		Double [] results = new Double[numberOfThreads]; 

		/*Takes start time for parallel power calculation*/
		startTime = System.currentTimeMillis();
		/*create power threads given by user*/
		for(int i = 0; i < numberOfThreads; i++ ){
			threads[i] = new PowerThread(i,number,eachPartValue,results);
			threads[i].start();
		}
				
		/*wait for all threads to complete*/
		for(int i = 0; i < numberOfThreads; i++ )
			threads[i].join();

		 /*initialze finalResult variable with default value 1.0 so we can used for multiply
                  with no. recived by other threads*/
		double finalResult = 1.0;
		/*calculate the final result by multipling thread's result */
		for(int i = 0; i < numberOfThreads; i++ )
    			finalResult*=results[i];
		System.out.println(number+" to the Power of "+power +" is "+finalResult);
		/*Takes end time for parallel power calculation*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in parallel  Power Calculation is :"+ (endTime-startTime)/(1000.0)+" Seconds");

		Double [] eresults = new Double[numberOfThreads]; 

		/*Takes the start time for parallel power calculation with Executor Service*/
		startTime = System.currentTimeMillis();
		/*Create the Executor service thread pool given by user*/
		ExecutorService e = Executors.newFixedThreadPool(numberOfThreads);
		final CountDownLatch doneSignal = new CountDownLatch(numberOfThreads);

		/*create method inner class for create PowerThread*/
		class PowerThread extends Thread{

			/*define instance variables*/
        		double number;
        		long power;
        		int threadNo;
        		Double []results;

			/*inner class constructor takes thread no.,number to which power to be calculated,
			power value, result array where result to be putted*/
        		public PowerThread(int threadNo,double number,long power,Double [] results){
                		this.number = number;
                		this.power = power;
                		this.threadNo = threadNo;
                		this.results = results;
        		}
			/* inner thread class overriden run method which call 
			PowerOfN class static method for power calculation*/
        		public void run (){
               	 		System.out.println(threadNo +" Thread Start "+number+" End "+power);
                		results[threadNo] = PowerOfN.calculatePower(number,power);
				doneSignal.countDown();

        		}
		}

		/*create threads for power claculation given by user*/
		for(int i = 0; i < numberOfThreads; i++ )
			e.execute(new PowerThread(i,number,eachPartValue,eresults));
		try {
                        doneSignal.await(); // wait for all to finish
                } catch (InterruptedException ie) {
                }
		/* shutdown the Executor service pool*/
                e.shutdown();
		/*calculate the final result by multipling thread's result */
		/*Reassign finalResult variable with default value 1.0 so we can used for multiply
                  with no. recived by other threads*/

		finalResult = 1.0;
		for(int i = 0; i < numberOfThreads; i++ )
    			finalResult*=eresults[i];
		/*Takes the end time for parallel power calculation with Executor Service*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in parallel concurrent Power Calculation is :"+ (endTime-startTime)/(1000.0)+" Seconds");

	}
}
