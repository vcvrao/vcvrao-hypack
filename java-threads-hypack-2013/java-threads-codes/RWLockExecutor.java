/********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Code Name            : RWLockExecutor.java 

 Objective            : Write a java concurrent APIs for concurrently read and write
                        a memory bolck using new "Read-write lock" (New Concurrency Java APIs)
                        than old "synchronization" construct.

 Input                : Number of Threads

 Output               : Execution time for Read-write lock/Synchronized/Read-Write
                        Lock with Executor API

 APIs Used            : Java Concurrent APIs -Read-Write Lock ,Executor API

 Created              : August-2013

 E-mail               : hpcfte@cdac.in     

********************************************************************************/

package concurrent;

/*import concurrent package and subpackage*/
import java.util.concurrent.*;
import java.util.concurrent.locks.*;

/*Thread class for reading data from array */
class ArrayRThread extends Thread{
	
	/*instance variable declaration*/	
	double [] pointToCommonArray;
	int threadNo;


	/*Thread class constructor taking thread no.,common array as arguments*/
	public ArrayRThread(int threadNo,double [] commonArray){
		pointToCommonArray = commonArray;
		this.threadNo = threadNo;
	}
	
	/*Thread class overriden run() method*/
	public void run (){
        	double temp ;

		/*Accessing data from common array more than 100,00,00,000 times
		using synchronization construct to allow only one thread 
		to execute following bolck of code*/
		synchronized(pointToCommonArray){
                	System.out.println(threadNo +" Read Thread Start ");
			for(int i =0; i<RWLockExecutor.nTimesOuterLoop;i++)
				for(int j =0; j<RWLockExecutor.nTimesInnerLoop;j++)
					temp = pointToCommonArray[1];

               	 	System.out.println(threadNo +" Read Thread  End ");
		}

	}
}

/*Thread class for writting data into Array*/
class ArrayWThread extends Thread{

	/*Instance variable declaration*/
        double [] pointToCommonArray = null;
        int threadNo = 0;

	/*Thread class constructor takes thread no.,common array as arguments*/
        public ArrayWThread(int threadNo,double [] commonArray){
                pointToCommonArray = commonArray;
                this.threadNo = threadNo;
        }

	/*Thread class overriden run() method*/
        public void run (){
                double temp ;

		/*Putting data into array more than 100,00,,00,000 times
		using synchronization construct to allow only 
		one thread to execute following block of code*/
                synchronized(pointToCommonArray){
                	System.out.println(threadNo +" Write Thread Start ");
			for(int i =0; i<RWLockExecutor.nTimesOuterLoop;i++)
				for(int j =0; j<RWLockExecutor.nTimesInnerLoop;j++)
					temp = pointToCommonArray[1];
                        pointToCommonArray[1]=2.0*(threadNo+1);
                	System.out.println(threadNo +" Write Thread  End ");
                }
        }
}


/*Main class*/
public class RWLockExecutor {

        /* static variable declartion used in Thread class run() method*/
        /*define outer for loop iteration, if value is increases
        then each thread takes more execution time*/
        static long nTimesOuterLoop = 1000000000;
        /*define inner for loop iteration, if value is increases
        then each thread takes more execution time*/
        static long nTimesInnerLoop =10;

	
	/*main() method takes array size and no. of threads as argument*/
	public static void main(String[] args) throws Exception{
		/*checks commandline argument validity*/
		if(args.length!=1){
			System.out.println("Invalid Argument <Number of threads>");
			return;
		}
	
	
		int dArraySize = 100;
		/*Takes number of Threads from commandline argument*/ 
		int numberOfThreads = Integer.parseInt(args[0].trim());
		
		/*Initialize the time variables */
		long startTime = 0;
		long endTime = 0;
		
		/* Create the intermediate arrays */
		double [] dArray = new double[dArraySize];
		double [] pArray = new double[dArraySize];
		double [] finalArray = null;
		
		/*fill the array elements in decending order  */
		for(int i = 0; i < dArraySize; i++){
			dArray[i]=dArraySize-i;
			pArray[i]=dArraySize-i;
			
		}
 
		
		Thread [] threads = new ArrayRThread[numberOfThreads];
		Thread [] wthreads = new ArrayWThread[numberOfThreads];
		
		/*Takes the start time for parallel read write operation 
		using synchronization construct */
		startTime = System.currentTimeMillis();
		
		/*create one write thread and no. of read threads given by user 
		using thread no.,common array as a arguments*/
		for(int i = 0; i < numberOfThreads; i++ ){
                        if(i == 0){
                                wthreads[i] = new ArrayWThread(i,pArray);
                                wthreads[i].start();
                        }

			threads[i] = new ArrayRThread(i,pArray);
			threads[i].start();
		}

		/* main thread waits for write thread */	
		for(int i = 0; i < 1; i++ )
			wthreads[i].join();
				
		/* main thread waits for all other threads */	
		for(int i = 0; i < numberOfThreads; i++ )
			threads[i].join();
		
		/*Takes the end time for parallel read write operation 
		using synchronization construct */
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution synchronized() is :"+ (endTime-startTime)/(1000.0)+" Seconds");
		
		/*Takes start time for common array read and write operations 
		using ReadWrite Lock*/
		startTime = System.currentTimeMillis();
		
		/*Declare new ReadWrite lock and get ReadLock & WriteLock from it*/
    		ReentrantReadWriteLock rwl = new ReentrantReadWriteLock();
		ReentrantReadWriteLock.ReadLock rl = rwl.readLock();	
    		ReentrantReadWriteLock.WriteLock wl = rwl.writeLock();	
		
		/*Thread class for reading data from array */
		class RLThread extends Thread{

			/*instance variable declaration*/
     			double [] pointToCommonArray;
        		int threadNo;
    			ReentrantReadWriteLock.ReadLock rw;	

			/*Thread class constructor taking thread no.,
			common array and ReadLock as arguments*/
        		public RLThread(int threadNo,double [] commonArray,ReentrantReadWriteLock.ReadLock rw){ 	
                		pointToCommonArray = commonArray;
                		this.threadNo = threadNo;
    				this.rw = rw;	
        		}
			
			/*Thread class overriden run() method*/
        		public void run (){
                		double temp ;

				/*Taking Read lock before reading data from shared array,
				More than one threads can take ReadLock simultaneously*/
				rw.lock();
				try{
                			System.out.println(threadNo +" ReadLock Thread Start ");
					/*Accessing data from common array more 
					than 100,00,00,000 times*/
					for(int i =0; i<RWLockExecutor.nTimesOuterLoop;i++)
						for(int j =0; j<RWLockExecutor.nTimesInnerLoop;j++)
							temp = pointToCommonArray[1];
               	 			System.out.println(threadNo +" ReadLock Thread  End ");
				}
        			finally { 
					/*Releasing Read lock */
					rw.unlock(); 
				}
                	}

        	}

		/*Thread class for writting data into Array*/
		class WLThread extends Thread{

			/*instance variable declaration*/
        		double [] pointToCommonArray;
        		int threadNo;
    			ReentrantReadWriteLock.WriteLock wl;	

			/*Thread class constructor taking thread no.,
			common array and WriteLock as arguments*/
        		public WLThread(int threadNo,double [] commonArray,ReentrantReadWriteLock.WriteLock rw){ 	
                		pointToCommonArray = commonArray;
                		this.threadNo = threadNo;
				this.wl = rw;
        		}
			
			/*Thread class overriden run() method*/
        		public void run (){
                		double temp ;

				/*Taking Write lock before writting data into shared array,
				only one threads can take WriteLock no ReadLock can be 
				taken by other threads */
				wl.lock();
				try
				{
                			System.out.println(threadNo +" WriteLock Thread Start ");
					for(int i =0; i<RWLockExecutor.nTimesOuterLoop;i++)
						for(int j =0; j<RWLockExecutor.nTimesInnerLoop;j++)
                        				pointToCommonArray[1]=2.0*(threadNo+1);
                			System.out.println(threadNo +" WriteLock Thread End ");
                		}
        			finally { 
					/*Releasing Write lock */
					wl.unlock(); 
				}
        		}
		}

		
		RLThread [] rlthreads = new RLThread[numberOfThreads]; 
		WLThread [] wlthreads = new WLThread[numberOfThreads]; 
		
		/*create one write thread and no. of read threads given by user 
		using thread no.,common array and ReadLock or WriteLock as a arguments*/
		for(int i = 0; i < numberOfThreads; i++ ){
			if(i == 0){
				wlthreads[i] = new WLThread(i,pArray,wl);
				wlthreads[i].start();
			}
			rlthreads[i] = new RLThread(i,pArray,rl);
			rlthreads[i].start();
				
		}
				
		/*wait for write thread to complete*/
		for(int i = 0; i < 1; i++ )
			wlthreads[i].join();
		
		/*wait for read threads to complete*/
		for(int i = 0; i < numberOfThreads; i++ )
			rlthreads[i].join();
		
		/*Takes end time for common array read and write operations 
		using ReadWrite Lock*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution readwritelock is :"+ (endTime-startTime)/(1000.0)+" Seconds");
                
		/*Takes start time for common array read and write operations 
		using ReadWrite Lock with Executor Service*/
		startTime = System.currentTimeMillis();
		
		/*Create a thread pool of no. of threads given by user */
		ExecutorService e = Executors.newFixedThreadPool(numberOfThreads+1);
		
		/*Create a CountDownLatchs with no of threads it has to 
		wait for read and write thread*/
                final CountDownLatch doneSignal = new CountDownLatch(numberOfThreads);
                final CountDownLatch doneSignalW = new CountDownLatch(1);
		
		/*Thread class for reading data from array */
                class RLThreadE extends Thread{
	
			/*instance variable declaration*/
                        double [] pointToCommonArray;
                        int threadNo = 0;
                        ReentrantReadWriteLock.ReadLock rw;

			/*Thread class constructor taking thread no.,
			common array and ReadLock as arguments*/
                        public RLThreadE(int threadNo,double [] commonArray,ReentrantReadWriteLock.ReadLock rw){
                                pointToCommonArray = commonArray;
                                this.threadNo = threadNo;
                                this.rw = rw;
                        }
						
			/*Thread class overriden run() method*/
                        public void run (){
                                double temp ;

				/*Taking Read lock before reading data from shared array.
				  More than one threads can take ReadLock simultaneously*/
                                rw.lock();
                                try{
                			System.out.println(threadNo +" ReadExecutor Thread Start ");
					/*Accessing data from common array more 
					than 100,00,00,000 times*/
                                        for(int i =0; i<RWLockExecutor.nTimesOuterLoop;i++)
                                        	for(int j =0; j<RWLockExecutor.nTimesInnerLoop;j++)
                                        		temp = pointToCommonArray[1];
               	 			System.out.println(threadNo +" ReadExecutor Thread  End ");
                                }
                                finally { 
					
					/*decrease the countDownLatch value by 1 */
					doneSignal.countDown(); 
					
					/*Releasing Read lock */
					rw.unlock(); 
				}
                	}

                }

		/*Thread class for writting data into Array*/
                class WLThreadE extends Thread{

			/*instance variable declaration*/
                        double [] pointToCommonArray;
                        int threadNo;
                        ReentrantReadWriteLock.WriteLock wl;

			/*Thread class constructor taking thread no.,
			common array and Write Lock as arguments*/
                        public WLThreadE(int threadNo,double [] commonArray,ReentrantReadWriteLock.WriteLock rw){
	                        pointToCommonArray = commonArray;
                                this.threadNo = threadNo;
                                this.wl = rw;
                        }
			
			/*Thread class overriden run() method*/
                        public void run (){
                                double temp ;

				/*Taking Write lock before writting data into shared array.
				only one threads can take WriteLock after this ReadLock cannot 
				be taken by any other threads */
                                wl.lock();
				try
                                {
                			System.out.println(threadNo +" WriteExecutor Thread Start ");
                                	for(int i =0; i<RWLockExecutor.nTimesOuterLoop;i++)
                                        	for(int j =0; j<RWLockExecutor.nTimesInnerLoop;j++)
                                			pointToCommonArray[1]=2.0*(threadNo+1);
                                	System.out.println(threadNo +" WriteExecutor Thread End ");
                                }
                                finally {  
					/*decrease the countDownLatch value by 1 */
					doneSignalW.countDown(); 
						
					/*Releasing Write lock */
					wl.unlock(); 
				}
			}
               } 
		/* assign the work to pool threads by create a new thread object 
		using thread no,common array ,write or read lock */
                for(int i = 0; i < numberOfThreads; i++ ){
			if(i==0){
                        	e.execute(new WLThreadE(i,pArray,wl));
			}
                        e.execute(new RLThreadE(i,pArray,rl));
		}
                try {
                        // wait for write thread to finish
                        doneSignalW.await(); 
                        // wait for all read thread to finish
                        doneSignal.await(); 
                } catch (InterruptedException ie) {
                }
		/*shut down the Executor service */
                e.shutdown();
		/*Takes end time for common array read and write operations 
		using ReadWrite Lock with Executor Service*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution readwritelock is with executor :"+ (endTime-startTime)/(1000.0)+" Seconds");
                

	}

}


