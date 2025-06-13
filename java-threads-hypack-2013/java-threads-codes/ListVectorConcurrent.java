/*******************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Code Name          : ListVectorConcurrent.java

 Objective        : Write a java concurrent APIs program for concurrently
                    read and write to a Vector and ArrayList collection and
                    check the performance improvement

 Input           : Number of Threads

 Output          : Execution time for Vector/ArrayList

 APIs Used       : - 

 Created         : August-2013

 E-mail          : hpcfte@cdac.in     

************************************************************************************/

package concurrent;

/*import java concurrent package */
import java.util.concurrent.*;
import java.util.*;

/*Thread class for read data from Vector */
class LRThread extends Thread{

	/*Instance variables declaration*/	
	Vector vector ;
	int threadNo;

	/*Thread class constructor takes thread no,
	common Vector as arguments*/
	public LRThread(int threadNo,Vector vector){
		this.vector = vector;
		this.threadNo = threadNo;
	}

	/*Thread class overriden run() method*/
	public void run (){
        	Integer temp;
            	System.out.println(threadNo +" Read Thread Start ");
		/*Accessing data from Vector 5,00,00,000 times*/
		for(int i =0; i<ListVectorConcurrent.nTimes;i++){
			if((temp = (Integer)vector.get(0))==null)
				System.out.println("Vector data not found "+temp );
		}
            	System.out.println(threadNo +" Read Thread  End ");

	}
}

/*Thread class for write data into Vector*/
class LWThread extends Thread{

	/*Instance variables declaration*/
	Vector vector ;
        int threadNo;

	/*Thread class constructor takes thread no.,common Vector*/
        public LWThread(int threadNo,Vector vector){
		this.vector = vector;
            	this.threadNo = threadNo;
        }

	/*Thread class overriden run() method*/
        public void run (){
                double temp ;
                System.out.println(threadNo +" Write Thread Start ");
		/*Putting data into Vector 1,00,000 times*/
		for(int i =0; i<ListVectorConcurrent.nTimesWriteOperation;i++){
			if(!vector.add(new Integer(i+1)))
                       		System.out.println("Vector: data not inserted "+i);
		}
                System.out.println(threadNo +" Write Thread  End ");
        }
}

/*Main class*/
public class ListVectorConcurrent {

        /*static variable declartion used in Thread class run() method*/
        /*define no. of loop iteration for read operation, if value is increases
        then each thread takes more execution time*/
        static long nTimes =500000 ;
        /*define no. of loop iteration for write operation, if value is increases
        then each thread takes more execution time*/
        static long nTimesWriteOperation =5000 ;

	/*main() method takes number of thread as a commandline argument*/	
	public static void main(String[] args) throws Exception{

		/*checks commandline argument validity*/
		if(args.length!=1){
			System.out.println("Invalid Argument <Number Of Threads>");
			return;
		}

		/*Takes number of threads from commandline argument*/
		int numberOfThreads = Integer.parseInt(args[0].trim());

		/*create Vector*/
		Vector vector = new Vector();
		/*Putting data into Vector*/
		if(vector.add(new Integer(0)))
			System.out.println("Vector data inserted "+vector.get(0));
		long startTime = 0;
		long endTime = 0;

		/*Takes start time for Vector read and write operations*/
		startTime = System.currentTimeMillis();

		/*create one write thread and no. of read threads given by user*/
		Thread [] lthreads = new LRThread[numberOfThreads];
		Thread [] lwthreads = new LWThread[numberOfThreads];
		for(int i = 0; i < numberOfThreads; i++ ){
               		if(i == 0){
				lwthreads[i] = new LWThread(i,vector);
                    		lwthreads[i].start();
			}

			lthreads[i] = new LRThread(i,vector);
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
		ArrayList arrayList = new ArrayList();

		/*Thread class for read from ArrayList*/
		class RAThread extends Thread{

			/*Instance variable declaration*/
        		int threadNo;
			ArrayList arrayList;
        		
			/*Thread class constructor takes thread no,
			common ArrayList as arguments*/
			public RAThread(int threadNo,ArrayList arrayList){ 	
                		this.arrayList = arrayList;
				this.threadNo = threadNo;
        		}

			/*Thread class overriden run() method*/
        		public void run (){
                		Integer temp= null;

                        	System.out.println("A Read Thread no. "+threadNo +"start ");
				/*Accessing data from ArrayList 5,00,00,000 times*/
				for(int i =0; i<ListVectorConcurrent.nTimes;i++){
					try{
					if((temp = (Integer)arrayList.get(0))==null)
						System.out.println("ArrayList: data not found "+temp );
					}catch(Exception e){System.out.println("ArrayList Exception "+temp );}
				}
                		System.out.println("A Read Thread  "+threadNo +" End ");
                	}

        	}

		/*Thread class for write into ArrayList*/
		class WAThread extends Thread{

			/*Instance variable declaration*/
			ArrayList arrayList;
        		int threadNo;
        		
			/*Thread class constructor takes thread no,
			common ArrayList as arguments*/
			public WAThread(int threadNo,ArrayList arrayList){ 	
                		this.threadNo = threadNo;
                		this.arrayList = arrayList;
        		}

			/*Thread class overriden run() method*/
        		public void run (){
                		Integer temp ;

                        	System.out.println("A write Thread no. "+threadNo +" start ");
				/*Putting data into ArrayList 1,00,000 times*/
				for(int i =0; i<ListVectorConcurrent.nTimesWriteOperation;i++){
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
		RAThread [] rlthreads = new RAThread[numberOfThreads]; 
		WAThread [] wlthreads = new WAThread[1]; 
		for(int i = 0; i < numberOfThreads; i++ ){
			if(i == 0){
				wlthreads[i] = new WAThread(i,arrayList);
				wlthreads[i].start();
			}
			rlthreads[i] = new RAThread(i,arrayList);
			rlthreads[i].start();
				
		}
				
		/*wait for write thread to complete*/
		for(int i = 0; i < 1; i++ )
			wlthreads[i].join();
		/*wait for read threads to complete*/
		for(int i = 0; i < numberOfThreads; i++ )
			rlthreads[i].join();

		/*Takes start time for ArrayList read and write operations*/
		endTime = System.currentTimeMillis();
    		System.out.println("Total elapsed time in execution ArrayList is :"+ (endTime-startTime)/(1000.0)+" Seconds");

	}

}


