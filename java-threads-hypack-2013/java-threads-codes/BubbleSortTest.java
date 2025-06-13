/***************************************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Code Name	    : BubbleSortTest.java

 Objective 	    : Write a java concurrent APIs program for sorting an 
		      un-sorted array using Buble Sorting Algorithm 
 
 Input		    : Double Array Length
		      and Number of Threads

 Output		    : Sorted Array with execution time for serial/parallel/Executor API 	                                            

 APIs Used          : Java Concurrent APIs -Executor API

 Created            : August-2013

 E-mail             : hpcfte@cdac.in     

*****************************************************************************************************/

package concurrent;

import java.util.concurrent.*;
/* Thread class for create a thread */
class BubbleThread extends Thread {
	
	/* Instance Variable declaration */
	double [] pointToCommonArray;
	int threadNo;
	int initialValue;
	int eachPartValue,threadArrayStart,threadArrayEnd =0;

	/* Thread class constructor takes thread no., array length which is sorted by 
	this thread,common array which is shared by each thread as arguments*/
	public BubbleThread(int threadNo,int eachPartValue,double [] commonArray){
		pointToCommonArray = commonArray;
		this.threadNo = threadNo;
		this.eachPartValue =eachPartValue;
		threadArrayStart =(threadNo*eachPartValue) ;
		threadArrayEnd   = ((threadNo+1)*eachPartValue-1); 		
	}

	/* Thread class overriden run() method */
	public void run (){
	        int  counter, index;
        	double temp ;
        	int length=threadArrayEnd-threadArrayStart+1;

		System.out.println(threadNo +" Thread Start "+threadArrayStart+" End "+threadArrayEnd);

		//Loop once for each element in the array.
		for(counter=threadArrayStart; counter<threadArrayEnd; counter++) { 
            		//Once for each element, minus the counter.
            		for(index=threadArrayStart; index<threadArrayStart+threadArrayEnd-counter; index++) { 
                		//Test if need a swap or not.
                		if(pointToCommonArray[index] > pointToCommonArray[index+1]) { 
                    			//These three lines just swap the two elements:
                    			temp = pointToCommonArray[index]; 
                    			pointToCommonArray[index] = pointToCommonArray[index+1];
                    			pointToCommonArray[index+1] = temp;
                		}
            		}
        	}

	}
}

/* Main class */
public class BubbleSortTest {
	
	/* Main() method which takes two commandline arguments Array Size 
	and number of threads */
	public static void main(String[] args) throws Exception{

		/* check the number of commandline argument */
		if(args.length !=2){
			System.out.println("Invalid Arguments <Array Size> <Number of threads>");
			return;
		}
	
		/* Intialize Array size and number of threads through commandline */
		int dArraySize = Integer.parseInt(args[0].trim());
		int numberOfThreads = Integer.parseInt(args[1].trim());

		/*check the Number of threads should 2 or 4 or 8 */
		if(!((numberOfThreads == 2)||(numberOfThreads == 4)||(numberOfThreads == 8))){
			System.out.println("Invalid Arguments <Array Size> <Number of threads>");
			System.out.println("Number of threads should 2 or 4 or 8 ");
			return;
		}

		/*check the Array size should be divisible by Number of threads */
		if((dArraySize % numberOfThreads)!=0){
			System.out.println("Invalid Arguments <Array Size> <Number of threads>");
			System.out.println("Array Size should divisible by number of threads ");
			return;
		}
	
		/* Calculate the Array length for each thread */
		int eachPartValue = dArraySize/numberOfThreads;

		Thread [] threads = new BubbleThread[numberOfThreads];

		/*Initialize the time variables */
		long startTime = 0;
		long endTime = 0;

		/* Create the intermediate arrays which is used for bubble sort */
		double [] dArray = new double[dArraySize];
		double [] pArray = new double[dArraySize];
		double [] fArray = new double[dArraySize];
		double [] tArray = new double[dArraySize];
		double [] qArray = new double[dArraySize];
		double [] finalArray = null;

		/*Initialize the arrays with decending order to get worst time complexity */
		for(int i = 0; i < dArraySize; i++){
			dArray[i]=dArraySize-i;
			pArray[i]=dArraySize-i;
			
		}

 		/* Calculate Max and Available Heap Memory */
       		System.out.println("Total Memory	"+Runtime.getRuntime().totalMemory()/(1024.0*1024.0*1024.0)+" GB");    
      	 	System.out.println("Max Memory	"+Runtime.getRuntime().maxMemory()/(1024.0*1024.0*1024.0)+" GB");
       		System.out.println("Free Memory	"+Runtime.getRuntime().freeMemory()/(1024.0*1024.0*1024.0)+" GB");

		/*Takes the start time for serial bubble sort */
		startTime = System.currentTimeMillis();

		/*call the bubblesort method with array and array size arguments*/
		bubbleSort(dArray,dArraySize);

		/*Takes the end time for serial bubble sort */
		endTime = System.currentTimeMillis();

    		System.out.println("Total elapsed time in execution serial bubbleSort() is :"+ (endTime-startTime)/(1000.0)+" Seconds");
		
		/*Takes the start time for parallel bubble sort */ 
		startTime = System.currentTimeMillis();

		/*create the threads with thread no.,length of array which is 
		used by particular thread and array */
		for(int i = 0; i < numberOfThreads; i++ ){
			threads[i] = new BubbleThread(i,eachPartValue,pArray);
			threads[i].start();
		}

		/* main thread waits for all other threads */	
		for(int i = 0; i < numberOfThreads; i++ )
			threads[i].join();

    		/*merge the results coming from diffrent threads */
		int i,j = 0;
 		for (i = 0; i <numberOfThreads; i+=2){
			int startIndex = (i*eachPartValue);
			int lastIndex= ((i+2)*eachPartValue-1);
			merge(startIndex,lastIndex,pArray,tArray);
		}

		/*Merge requires again if more than 2 threads are there*/
		if(i >2){
			for (j = 0; j < numberOfThreads; j+=4){
				int startIndex = j*eachPartValue;
				int lastIndex  = ((j+4)*eachPartValue)-1;	
				merge(startIndex,lastIndex,tArray,fArray);
			}
		}
		
		/*Merge requires again and assign to resultant array 
		if more than 4 threads are there*/
		if(j > 4){
			merge(0,pArray.length-1,fArray,qArray);
			finalArray=qArray;
		}

		/*assign to resultant array 2 threads are there*/
		else if(j == 0){ 
			finalArray=tArray;
		}

		/*assign to resultant array 4 threads are there*/
		else if(j==4){
			finalArray=fArray;
		}

		/*Takes the end time for parallel bubble sort */ 
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution parallel bubbleSort is :"+ (endTime-startTime)/(1000.0)+" Seconds");

		/*compare the serial sorted array with parallel sorted array 
		if not match then sorting is not done properly */ 
		for (i = 0; i < pArray.length; i++) {
			if (dArray[i]!=finalArray[i]) {
    			System.out.println("Not sorted");
			break;
			}
		}

		/*Takes the start time for parallel bubble sort 
		using Executor (Java Concurrent API)*/ 
                startTime = System.currentTimeMillis();

		/*Create a thread pool of no. of threads given by user */
                ExecutorService e = Executors.newFixedThreadPool(numberOfThreads);

		/*Create a CountDownLatch with no of threads it has to wait */
                final CountDownLatch doneSignal = new CountDownLatch(numberOfThreads);

		/*create a method innner class for create a thread*/
		class BubbleThread extends Thread{

			/* Instance Variable declaration */
        		double [] pointToCommonArray;
        		int threadNo;
        		int initialValue;
        		int eachPartValue,threadArrayStart,threadArrayEnd =0;

			/* Thread class constructor */
        		public BubbleThread(int threadNo,int eachPartValue,double [] commonArray){
                		pointToCommonArray = commonArray;
                		this.threadNo = threadNo;
                		this.eachPartValue =eachPartValue;
                		threadArrayStart =(threadNo*eachPartValue) ;
                		threadArrayEnd   = ((threadNo+1)*eachPartValue-1);
        		}

			/* Thread class overriden run() method */
        		public void run (){
                		int  counter, index;
                		double temp ;
                		int length=threadArrayEnd-threadArrayStart+1;

                		System.out.println(threadNo +" Thread Start "+threadArrayStart+" End "+threadArrayEnd);
                                 //Loop once for each element in the array.
                                for(counter=threadArrayStart; counter<threadArrayEnd; counter++) { 
                                       	//Once for each element, minus the counter.
                                       	for(index=threadArrayStart; index<threadArrayStart+threadArrayEnd-counter; index++) { 
                                               	//Test if need a swap or not.
                                               	if(pointToCommonArray[index] > pointToCommonArray[index+1]) { 
                                                       	//These three lines just swap the two elements:
                                                       	temp = pointToCommonArray[index];
                                                       	pointToCommonArray[index] = pointToCommonArray[index+1];
                                                       	pointToCommonArray[index+1] = temp;
                                               	}
                                       	}
                                }
				/*decrease the countDownLatch value by 1 */
				doneSignal.countDown();	
        		}
		}

		/* Create the intermediate arrays which is used for sort */
                double [] pArrayC = new double[dArraySize];
                double [] fArrayC = new double[dArraySize];
                double [] tArrayC = new double[dArraySize];
                double [] qArrayC = new double[dArraySize];
                double [] finalArrayC = null;

		/*Initialize the arrays with decending order to get worst time complexity */
                for(int k = 0; k < dArraySize; k++){
                        pArrayC[k]=dArraySize-k;

                }

		/* assign the work to pool threads by create a new thread object using 
		thread no,array size which is sort by this thread,array */
                for(int k = 0; k < numberOfThreads; k++ )
                        e.execute(new BubbleThread(k,eachPartValue,pArrayC));
                	try {
                        	doneSignal.await(); // wait for all to finish
                	} catch (InterruptedException ie) {
                }

		/*shutdown the Executor service */
                e.shutdown();

    		/*merge the sorted results coming from diffrent threads */
                int k,l = 0;
                for (k = 0; k <numberOfThreads; k+=2){
                        int startIndex = (k*eachPartValue);
                        int lastIndex= ((k+2)*eachPartValue-1);
                        merge(startIndex,lastIndex,pArrayC,tArrayC);
                }

		/*Merge requires again if more than 2 threads are there*/
                if(k >2){
                        for (l = 0; l < numberOfThreads; l+=4){
                                int startIndex = l*eachPartValue;
                                int lastIndex  = ((l+4)*eachPartValue)-1;
                                merge(startIndex,lastIndex,tArrayC,fArrayC);
                        }
                }

		/*Merge requires again and assign to resultant array 
		if more than 4 threads are there*/
                if(l > 4){
                        merge(0,pArrayC.length-1,fArrayC,qArrayC);
                        finalArrayC=qArrayC;
                }

		/*assign to resultant array 2 threads are there*/
                else if(l == 0){
                        finalArrayC=tArrayC;
                }

		/*assign to resultant array 4 threads are there*/
                else if(l==4){
                        finalArrayC=fArrayC;
                }
		/*Takes the end time for parallel bubble sort using Executor (Java Concurrent API)*/ 
                endTime = System.currentTimeMillis();

                System.out.println("Total elapsed time in concurrent parallel BubbleSort is :"+ (endTime-startTime)/(1000.0)+" Seconds");

		/*compare the serial sorted array with parallel sorted array 
		if not match then sorting is not done properly */ 
                for (i = 0; i < pArrayC.length; i++) {
                        if (dArray[i]!=finalArrayC[i]) {
                        	System.out.println("Not sorted");
                        	break;
                        }
                }

	}

	/* static method for merge the two sorted subpart 
	of the array and assign to fArray */
	public static void merge(int start,int end,double[] pArray,double[] fArray){
		end +=1;

		/*calculate the mid position of the array */
		int mid = (end+start)/2;
	
		/* itreate the loop up to end of the array */
		for (int i =start,p = start, q = mid; i< end; i++){

			/*compare the array's subpart element and assign to the fArray */
			if(q >=end || p < mid && (pArray[p] <= pArray[q])){
				fArray[i] = pArray[p++];
			}

			else
			fArray[i] = pArray[q++];
		}
	}

	/* static method for print the Array elements */
	public static void printArray(int name,double [] array){
		System.out.println(name+ " Array Elements : ");
		for(int i = 0; i < array.length; i++)
    			System.out.print(" "+array[i]);
    		System.out.println(" ");
	}

	/* static method for serial bubble sort */
    	public  static void bubbleSort(double[] unsortedArray, int length) {

        	int  counter=0, index=0;
        	double temp = 0;

       	 	//Loop once for each element in the array.
       	 	for(counter=0; counter<length-1; counter++) { 
            		//Once for each element, minus the counter.
            		for(index=0; index<length-1-counter; index++) { 
                		//Test if need a swap or not.
                		if(unsortedArray[index] > unsortedArray[index+1]) { 
                    			//These three lines just swap the two elements:
                    			temp = unsortedArray[index]; 
                    			unsortedArray[index] = unsortedArray[index+1];
                    			unsortedArray[index+1] = temp;
                		}
            		}
        	}
    	}
}


