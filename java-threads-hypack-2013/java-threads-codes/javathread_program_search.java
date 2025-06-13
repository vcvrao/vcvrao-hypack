/***********************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 1.2 : javathread_program_search.java
 
  Objective   : To search the minimum number in a given array. 
                and Demostrate use of 
			Runnable interface
			Thread()
			run()

  Input       : Array size
       		Number of thread

  Output      : Minimum element in a array 
       		Execution Time

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************************/

import java.util.*;
public class javathread_program_search implements Runnable
{
         /* declare variable */
	 int eSize,numThread;
	 double min ;
         int dist =0;
	 long startTime,endTime,diffTime=0;
	 double [] element;
	 Thread [] t;
	/*main method start here */

	 public static void main(String args[])
	{

	 	int eSize1,numThread1;
		/* check for correct number of argument */

		if(args.length != 2)
		{
			System.out.println(" Invalid Number of Argument It Must Be Two ");
			return ;
		}
		 
		eSize1 = Integer.parseInt(args[0]);
		numThread1 = Integer.parseInt(args[1]);

		/* check no of thread could not more then 8 */

		if( numThread1 > 8)
		{
			System.out.println("no of thread could not more then 8 ");
			 return ;
		}
		
		/* check for array size is factor of number of thread */

		if(eSize1 % numThread1 != 0)
		{
			System.out.println("array size is not a factor of number of thread ");
			 return ;
		}
		
		javathread_program_search m = new javathread_program_search();
		m.read_array(eSize1,numThread1);
		m.serial_search(eSize1);

	}
	/* read array function start here */

	public void  read_array(int eSize,int numThread)
	{
		this.eSize = eSize;
		this.numThread = numThread;
		element= new double[eSize];
		int counter;
		Random rn=new Random();

		/* initilizing the array */

		for(counter = 0; counter < eSize ; ++counter)
		{
			element[counter] = rn.nextDouble();
	//		System.out.print(element[counter] + " ");
		}

		dist = eSize / numThread;
		min=element[0];

		creat_thread();
                for(counter = 0; counter < numThread ; ++ counter)
                {
                        try
                        {
                                t[counter].join();
                        }
                        catch( InterruptedException e)
                        {
                                System.out.println(" Main Thread interrupted ");
                        }
		}

		endTime = System.currentTimeMillis();
		diffTime= endTime - startTime;
		System.out.println("");
		System.out.printf(" Minimum Elelment Is %5.3f ",min);		
		System.out.println(" total time in millisecond   " + diffTime);		
	}
	
	/* creat thread methode start here */	

	public void creat_thread()
	{
		int counter;
		t = new Thread[numThread];
		startTime = System.currentTimeMillis();
                for(counter = 0; counter < numThread ; ++ counter)
                {
                                t[counter] = new Thread(this,""+counter);
                                t[counter].start();
                }

	}
	
	/*function for searial search */
	
	public void serial_search(int eSize)
	{

		double cMin;
		int counter;
                this.eSize = eSize;
                cMin = element[0];
		long startTime1,endTime1,diffTime1=0;
		startTime1 = System.currentTimeMillis();
		for(counter = 1; counter < eSize ; ++counter)
		{
			if( cMin > element[counter])
			{
				cMin = element[counter];
			}
		}
		endTime1 = System.currentTimeMillis();
		diffTime1= endTime1 - startTime1;
		System.out.printf("serial minimum element is %5.3f ",cMin);
		System.out.println(" total time for serial search in millisecond  " + diffTime1);		
	}

	
 /* call back methode run start here */

	public void run()
	{
		Thread tt = Thread.currentThread();

		int no=Integer.parseInt(tt.getName()) + 1;
		double localMin;
		int counter;
	        localMin = element[(no-1)* dist]; 			
		/* logic  to find the local minimum number */

		for(counter =((no -1) * dist); counter <= ((no * dist) -1) ; ++counter)
		{
			if(localMin > element[counter])
			{
				localMin = element[counter];
			}
		}
		/* synchronize block to find global minimum */
		synchronized(this)
		{
			if(min > localMin)
				min=localMin;
		}
	}

        


}


