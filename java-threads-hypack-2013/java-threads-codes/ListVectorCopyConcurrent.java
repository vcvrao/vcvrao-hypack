/****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Code Name          : ListVectorCopyConcurrent.java

 Objective          : Write a java concurrent APIs program for CopyOnwriteArrayList
                      collection which is thread safe version of old ArrayList collection,
                      allows concurrent updation on ArrayList while iterating

 Input              : Number of Threads

 Output             : Execution time for Vector/ArrayList/CopyOnWriteArrayList

 APIs Used          : Java Concurrent APIs - CopyOnWriteArrayList API

 Created            : August-2013

 E-mail             : hpcfte@cdac.in     

**************************************************************************/

package concurrent;

/*import java concurrent package */
import java.util.concurrent.*;
import java.util.*;

/*Thread class for read data from Vector */
class VRThread extends Thread{
	
	/*Instance variables declaration*/	
	Vector vector ;
	int threadNo;
	
	/*Thread class constructor takes thread no,common Vector as arguments*/
	public VRThread(int threadNo,Vector vector){
		this.vector = vector;
		this.threadNo = threadNo;
	}

	/*Thread class overriden run() method*/
	public void run (){
        	Integer temp;
            	System.out.println(threadNo +" Read Thread Start ");
		/*create a iterator*/
		Iterator itr = vector.iterator();
		try{
		Thread.sleep(ListVectorCopyConcurrent.sleepInterval);
		/*Accessing data from Vector using iterator 5,00,000 times*/
		for(int i =0; i<ListVectorCopyConcurrent.nTimes;i++){
			itr.next();
		}
		}catch(InterruptedException e){
			System.out.println("Not able to iterate b'coz "+e);
		}catch(ConcurrentModificationException e){
			System.out.println("Not able to iterate b'coz "+e);
		}
            	System.out.println(threadNo +" Read Thread  End ");

	}
}

/*Thread class for write data into Vector*/
class VWThread extends Thread{

	/*Instance variables declaration*/
	Vector vector ;
        int threadNo;

	/*Thread class constructor takes thread no.,common Vector*/
        public VWThread(int threadNo,Vector vector){
		this.vector = vector;
            	this.threadNo = threadNo;
        }

	/*Thread class overriden run() method*/
        public void run (){
                System.out.println(threadNo +" Write Thread Start ");
		/*Putting data into Vector 5,00,000 times*/
		for(int i =0; i<ListVectorCopyConcurrent.nTimes;i++){
			if(!vector.add(new Integer(i+1)))
                       		System.out.println("Vector: data not inserted "+i);
		}
                System.out.println(threadNo +" Write Thread  End ");
        }
}


/*Main class*/
public class ListVectorCopyConcurrent {

        /*static variable declartion used in Thread class run() method*/
        /*define no. of loop iteration, if value is increases
        then each thread takes more execution time*/
        static long nTimes =500000 ;
	/*define Thread sleep time if increases thread has to wait more time */
        static long sleepInterval =5000 ;

	
	/*main() method takes number of thread as a commandline argument*/	
	public static void main(String[] args) throws Exception{

		/*checks commandline argument validity*/
		if(args.length!=1){
			System.out.println("Invalid Argument <Number Of Threads>");
			return;
		}

		/*Takes number of threads from commandline argument*/
		int numberOfThreads = Integer.parseInt(args[0].trim());

		/*create a Vector*/
		Vector vector = new Vector();
		/*Putting data into Vector*/
		if(vector.add(new Integer(0)))
			System.out.println("Vector: data inserted "+vector.get(0));
		long startTime = 0;
		long endTime = 0;
		
		/*Takes start time for Vector read and write operations*/
		startTime = System.currentTimeMillis();
		Thread [] lthreads = new VRThread[numberOfThreads];
		Thread [] lwthreads = new VWThread[1];

		/*create one write thread and no. of read threads given by user*/
		for(int i = 0; i < numberOfThreads; i++ ){
               		if(i == 0){
				lwthreads[i] = new VWThread(i,vector);
                    		lwthreads[i].start();
			}

			lthreads[i] = new VRThread(i,vector);
			lthreads[i].start();
          
		}
		/*wait for write thread to complete*/
		for(int i = 0; i < 1; i++ )
			lwthreads[i].join();
				
		/*wait for read threads to complete*/
		for(int i = 0; i < numberOfThreads; i++ )
			lthreads[i].join();

		/*Takes start time for Vector read and write operations*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution Vector is :"+ (endTime-startTime)/(1000.0)+" Seconds");

		/*Takes start time for ArrayList read and write operations*/
		startTime = System.currentTimeMillis();
		/*create ArrayList*/
		ArrayList arrayList = new ArrayList();

		/*Thread class for read from ArrayList*/
		class ARThread extends Thread{

			/*Instance variable declaration*/
        		int threadNo = 0;
			ArrayList arrayList;
        		
			/*Thread class constructor takes thread no,
			common ArrayList as arguments*/
			public ARThread(int threadNo,ArrayList arrayList){ 	
                		this.arrayList = arrayList;
				this.threadNo = threadNo;
        		}

			/*Thread class overriden run() method*/
        		public void run (){
                		Integer temp;
                        	System.out.println("A Read Thread no. "+threadNo +"start ");
				/*create a iterator*/
				Iterator iter = arrayList.iterator();
				try{
				Thread.sleep(ListVectorCopyConcurrent.sleepInterval);
				/*Accessing data from ArrayList 
				5,00,00,000 times using iterator*/
				for(int i =0; i<ListVectorCopyConcurrent.nTimes;i++){
					iter.next();
				}
				}catch(InterruptedException e){
					System.out.println("Not able to iterate b'coz "+e);
				}catch(ConcurrentModificationException e){
					System.out.println("Not able to iterate b'coz "+e);
				}
                		System.out.println("A Read Thread  "+threadNo +" End ");
                	}

        	}

		/*Thread class for write into ArrayList*/
		class AWThread extends Thread{

			/*Instance variable declaration*/
			ArrayList arrayList;
        		int threadNo;
        		
			/*Thread class constructor takes thread no,
			common ArrayList as arguments*/
			public AWThread(int threadNo,ArrayList arrayList){ 	
                		this.threadNo = threadNo;
                		this.arrayList = arrayList;
        		}

			/*Thread class overriden run() method*/
        		public void run (){
                        	System.out.println("A write Thread no. "+threadNo +" start ");
				/*Putting data into ArrayList 5,00,000 times*/
				for(int i =0; i<ListVectorCopyConcurrent.nTimes;i++){
					if(!(arrayList.add(new Integer(i+1))))
						System.out.println("ArrayList: data not inserted "+i);
				}
                        	System.out.println("A write Thread no. "+threadNo +" End ");
			    }
		}

		/*Putting data into ArrayList */
		if((arrayList.add(new Integer(0))))
			System.out.println("inserted "+arrayList.get(0));

		/*create one write thread and no. of read threads given by user*/
		ARThread [] rlthreads = new ARThread[numberOfThreads]; 
		AWThread [] wlthreads = new AWThread[1]; 
		for(int i = 0; i < numberOfThreads; i++ ){
			if(i == 0){
				wlthreads[i] = new AWThread(i,arrayList);
				wlthreads[i].start();
			}
			rlthreads[i] = new ARThread(i,arrayList);
			rlthreads[i].start();
				
		}
				
		/*wait for write thread to complete*/
		for(int i = 0; i < 1; i++ )
			wlthreads[i].join();
		/*wait for read threads to complete*/
		for(int i = 0; i < numberOfThreads; i++ )
			rlthreads[i].join();

		/*Takes end time for ArrayList read and write operations*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution ArrayList is :"+ (endTime-startTime)/(1000.0)+" Seconds");

		/*Takes start time for CopyOnWriteArrayList read and write operations*/
		startTime = System.currentTimeMillis();
		/*create CopyonArrayList */
		CopyOnWriteArrayList cArrayList = new CopyOnWriteArrayList();

		/*Thread class for read from CopyOnWriteArrayList*/
		class RHThread extends Thread{

			/*Instance variable declaration*/
        		int threadNo;
			CopyOnWriteArrayList cArrayList;

			/*Thread class constructor takes thread no,
			common CopyOnWriteArrayList as arguments*/
        		public RHThread(int threadNo,CopyOnWriteArrayList cArrayList){ 	
				this.cArrayList	= cArrayList;	
				this.threadNo = threadNo;
        		}

			/*Thread class overriden run() method*/
        		public void run (){
                        	System.out.println("C Read Thread no. "+threadNo +"start ");
				/*create a iterator*/
				Iterator iter = cArrayList.iterator();
				try{
				Thread.sleep(ListVectorCopyConcurrent.sleepInterval);
				/*Accessing data from CopyOnWriteArrayList 
				5,00,00,000 times using iterator*/
				for(int i =0; i<ListVectorCopyConcurrent.nTimes;i++){
					iter.next();
				}
				}catch(NoSuchElementException e){
					System.out.println("Not able to iterate b'coz "+e);
				}catch(InterruptedException e){
					System.out.println("Not able to iterate b'coz "+e);
				}catch(ConcurrentModificationException e){
					System.out.println("Not able to iterate b'coz "+e);
				}
                		System.out.println("C Read Thread  "+threadNo +" End ");
                	}

        	}

		/*Thread class for write into CopyOnWriteArrayList*/
		class WHThread extends Thread{

			/*Instance variable declaration*/
        		int threadNo;
        		CopyOnWriteArrayList cArrayList;

			/*Thread class constructor takes thread no,
			common CopyOnWriteArrayList as arguments*/
			public WHThread(int threadNo,CopyOnWriteArrayList cArrayList){ 	
                		this.threadNo = threadNo;
                		this.cArrayList=cArrayList;
        		}

			/*Thread class overriden run() method*/
        		public void run (){
                		
                   		System.out.println("C write Thread no. "+threadNo +" start ");
				/*Putting data into CopyOnWriteArrayList 5,000 times*/
				for(int i =0; i<ListVectorCopyConcurrent.nTimes;i++){
					if(!(cArrayList.add(new Integer(i+1))))
						System.out.println("CopyOnWriteArrayList: data not inserted "+i);
				}
                        	System.out.println("C write Thread no. "+threadNo +" End ");
			}
		}

		/*Putting data into CopyOnWriteArrayList 5,000 times*/
		if(!(cArrayList.add(new Integer(0))))
			System.out.println("inserted "+cArrayList.get(0));

		/*create one write thread and no. of read threads given by user*/
		RHThread [] rhthreads = new RHThread[numberOfThreads]; 
		WHThread [] whthreads = new WHThread[1]; 
		for(int i = 0; i < numberOfThreads; i++ ){
			if(i == 0){
				whthreads[i] = new WHThread(i,cArrayList);
				whthreads[i].start();
			}
			rhthreads[i] = new RHThread(i,cArrayList);
			rhthreads[i].start();
				
		}
				
		/*wait for write thread to complete*/
		for(int i = 0; i < 1; i++ )
			whthreads[i].join();
		/*wait for read threads to complete*/
		for(int i = 0; i < numberOfThreads; i++ )
			rhthreads[i].join();
		/*Takes end time for CopyOnWriteArrayList read and write operations*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution CopyOnWriteArrayList is :"+ (endTime-startTime)/(1000.0)+" Seconds");
                		
	}

}


