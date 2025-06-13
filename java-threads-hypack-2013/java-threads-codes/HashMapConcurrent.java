/***************************************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Code Name           : HashMapConcurrent.java

 Objective           : Write a java concurrent APIs program for concurrently read
                       and write to a concurrentHashMap collection(New Concurrency Java APIs)
                       and compare performance with old Hashtable collection.

 Input               : Number of Threads

 Output              : Execution time for Hashtable/ConcurrentHashMap/HashMap
                                  
 APIs Used           : Java Concurrent APIs - ConcurrentHashMap API

 Created             : August-2013

 E-mail              : hpcfte@cdac.in     

*****************************************************************************************************/

package concurrent;

/*import java concurrent package & subpackage*/
import java.util.concurrent.*;
import java.util.*;

/*Thread class for reading data from Hashtable */
class RThread extends Thread{

	/*Instance variable declaration*/
	Hashtable hashtable;
	int threadNo;

	/*Thread class constructor taking thread no.,
	common Hashtable as arguments*/
	public RThread(int threadNo,Hashtable hashtable){
		this.hashtable = hashtable;
		this.threadNo = threadNo;
	}

	/*Thread class overriden run() method*/
	public void run (){
        	String temp ;
            	System.out.println(threadNo +" Read Thread Start ");
		/*Accessing data from Hashtable 50,00,000 times*/
		for(int i =0; i<HashMapConcurrent.nTimes;i++){
			/*check for accessing data*/
			if((temp = (String)hashtable.get(new Integer(0)))==null)
				System.out.println("Hashtable data not found ");
		}
            	System.out.println(threadNo +" Read Thread  End ");

	}
}

/*Thread class for writting data to Hashtable*/
class WThread extends Thread{

	/*Instance variable declaration*/
	Hashtable hashtable;
        int threadNo;

	/*Thread class constructor takes thread no.,
	common Hashtable as arguments*/
        public WThread(int threadNo,Hashtable hashtable){
		this.hashtable = hashtable;
            	this.threadNo = threadNo;
        }

	/*Thread class overriden run() method*/
        public void run (){
		System.out.println(threadNo +" Write Thread Start ");
		/*Putting data into Hashtable 1,00,000 times*/
		for(int i =0; i<HashMapConcurrent.nTimesWriteOperation;i++){
			/*check for putting data*/
			if((hashtable.put(new Integer(i+1), "second"))!=null)
                        	System.out.println("Hashtable data not inserted "+i);
		}
                System.out.println(threadNo +" Write Thread  End ");
        }
}

/*Main class*/
public class HashMapConcurrent {

        /*static variable declartion used in Thread class run() method*/
        /*define no. of loop iteration for read operation, if value is increases
        then each thread takes more execution time*/
        static long nTimes =50000000;
        /*define no. of loop iteration for write operation, if value is increases
        then each thread takes more execution time*/
        static long nTimesWriteOperation =100000 ;

	/*main() method takes no. of thread as argument*/
	public static void main(String[] args) throws Exception{

		/*checks commandline argument validity*/
		if(args.length!=1){
			System.out.println("Invalid Argument <Number of threads>");
			return;
		}
		/*Takes number of Threads from commandline argument*/ 
		int numberOfThreads = Integer.parseInt(args[0].trim());
		Thread [] threads = new RThread[numberOfThreads];

		/*create Hashtable Integer as a key and String as a value*/
		Hashtable <Integer,String> hashtable = new Hashtable <Integer,String>();

		/*put data into Hashtable */
		if((hashtable.put(new Integer(0),"first"))==null)
			System.out.println("Hashtable data inserted "+hashtable.get(new Integer(0)) );
		long startTime = 0;
		long endTime = 0;
		
		Thread [] wthreads = new WThread[numberOfThreads];

		/*Takes start time for Hashtable read and write operations*/
		startTime = System.currentTimeMillis();

		/*create one write thread and no. of read threads given by user*/
		for(int i = 0; i < numberOfThreads; i++ ){
               		if(i == 0){
				wthreads[i] = new WThread(i,hashtable);
                    		wthreads[i].start();
			}

			threads[i] = new RThread(i,hashtable);
			threads[i].start();
          
		}
		/*wait for write thread to complete*/
		for(int i = 0; i < 1; i++ )
			wthreads[i].join();
		/*wait for read threads to complete*/
		for(int i = 0; i < numberOfThreads; i++ )
			threads[i].join();

		/*Takes end time for Hashtable read and write operations*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution Hashtable is :"+ (endTime-startTime)/(1000.0)+" Seconds");

		/*Takes start time for ConcurrentHashMap read and write operations*/
		startTime = System.currentTimeMillis();

		/*create ConcurrentHashMap */
		ConcurrentHashMap chashmap = new ConcurrentHashMap();

		/*Thread class for reading data from ConcurrentHashMap*/
		class RLThread extends Thread{

			/*Instance variable declaration*/
        		int threadNo;
			ConcurrentHashMap chashmap;

			/*Thread class constructor takes thread no. and 
			common ConcurrentHashMap as agruments*/
        		public RLThread(int threadNo,ConcurrentHashMap chashmap){ 	
                		this.chashmap = chashmap;
				this.threadNo = threadNo;
        		}
			/*Thread class overriden run() method*/
        		public void run (){
                		String temp ;
                        	System.out.println("C Read Thread no. "+threadNo +"start ");
				/*Accessing data from ConcurrentHashMap 50,00,000 times*/
				for(int i =0; i<HashMapConcurrent.nTimes;i++){
					if((temp = (String)chashmap.get(new Integer(0)))==null)
						System.out.println("ConcurrentHashMap data not found "+temp );
				}
                		System.out.println("C Read Thread  "+threadNo +" End ");
                	}

        	}
		/*Thread class for writting data into ConcurrentHashMap*/
		class WLThread extends Thread{
			/*Instance variable declaration*/
			ConcurrentHashMap chashmap;
        		int threadNo;
        		/*Thread class constructor takes thread no., 
			common ConcurrentHashMap as arguments*/
			public WLThread(int threadNo,ConcurrentHashMap chashmap){ 	
                		this.threadNo = threadNo;
                		this.chashmap = chashmap;
        		}

			/*Thread class overriden run() method*/
        		public void run (){
                        	System.out.println("C write Thread no. "+threadNo +" start ");
				/*putting data into ConcurrentHashMap 1,00,000 times*/
				for(int i =0; i<HashMapConcurrent.nTimesWriteOperation;i++){
					if((chashmap.put(new Integer(i+1), "second"))!=null)
						System.out.println("ConcurrentHashMap data  not inserted "+i);
				}
                        	System.out.println("C write Thread no. "+threadNo +" End ");
			}
		}

		/*put data into ConcurrentHashMap */
		if((chashmap.put(new Integer(0),"first"))==null)
			System.out.println("ConcurrentHashMap data inserted "+chashmap.get(new Integer(0)) );
		
		/*create one write thread and no. of read threads given by user*/
		RLThread [] rlthreads = new RLThread[numberOfThreads]; 
		WLThread [] wlthreads = new WLThread[1]; 
		for(int i = 0; i < numberOfThreads; i++ ){
			if(i == 0){
				wlthreads[i] = new WLThread(i,chashmap);
				wlthreads[i].start();
			}
			rlthreads[i] = new RLThread(i,chashmap);
			rlthreads[i].start();
				
		}
		/*wait for write thread to complete*/		
		for(int i = 0; i < 1; i++ )
			wlthreads[i].join();
		/*wait for read threads to comlete*/
		for(int i = 0; i < numberOfThreads; i++ )
			rlthreads[i].join();

		/*Takes end time for ConcurrentHashMap read and write operations*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution ConcurrentHashmap is :"+ (endTime-startTime)/(1000.0)+" Seconds");

		/*Takes start time for HashMap read and write operations*/
		startTime = System.currentTimeMillis();
		/*create HaspMap*/
		HashMap hashmap = new HashMap();

		/*Thread class for reads data from HashMap*/
		class RHThread extends Thread{

			/*Instance variable declaration*/
        		int threadNo = 0;
			HashMap hashmap;

        		/*Thread class constructor takes thread no., 
			common HashMap as arguments*/
        		public RHThread(int threadNo,HashMap hashmap){ 	
                		this.hashmap = hashmap;
				this.threadNo = threadNo;
        		}

			/*Thread class overriden run() method*/
        		public void run (){
                		String temp ;
                        	System.out.println("H Read Thread no. "+threadNo +"start ");
				/*Accessing data from HashMap 50,00,000 times*/
				for(int i =0; i<HashMapConcurrent.nTimes;i++){
					if((temp = (String)hashmap.get(new Integer(0)))==null)
						System.out.println("HashMap data not found "+temp );
				}
                		System.out.println("H Read Thread  "+threadNo +" End ");
                	}

        	}

		/*Thread class for writting data into HashMap*/
		class WHThread extends Thread{

			/*Instance variable declaration*/
			HashMap hashmap;
        		int threadNo;
        		
        		/*Thread class constructor takes thread no., 
			common HashMap as arguments*/
			public WHThread(int threadNo,HashMap hashmap){ 	
                		this.threadNo = threadNo;
                		this.hashmap = hashmap;
        		}

			/*Thread class overriden run() method*/
        		public void run (){
                   		System.out.println("H write Thread no. "+threadNo +" start ");
				/*putting data into HashMap 1,00,000 times*/
				for(int i =0; i<HashMapConcurrent.nTimesWriteOperation;i++){
					if((hashmap.put(new Integer(i+1), "second"))!=null)
						System.out.println("HashMap data not inserted "+i);
				}
				System.out.println("H write Thread no. "+threadNo +" End ");
			}
		}

		/*put data into HashMap */
		if((hashmap.put(new Integer(0),"first"))==null)
			System.out.println("HashMap data inserted "+hashmap.get(new Integer(0)) );

		/*create one write thread and no. of read threads given by user*/
		RHThread [] rhthreads = new RHThread[numberOfThreads]; 
		WHThread [] whthreads = new WHThread[numberOfThreads]; 
		for(int i = 0; i < numberOfThreads; i++ ){
			if(i == 0){
				whthreads[i] = new WHThread(i,hashmap);
				whthreads[i].start();
			}
			rhthreads[i] = new RHThread(i,hashmap);
			rhthreads[i].start();
				
		}
				
		/*wait for write thread to complete*/
		for(int i = 0; i < 1; i++ )
			whthreads[i].join();
		/*wait for read threads to complete*/
		for(int i = 0; i < numberOfThreads; i++ )
			rhthreads[i].join();

		/*Takes end time for HashMap read and write operations*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution HashMap is :"+ (endTime-startTime)/(1000.0)+" Seconds");
	}
}


