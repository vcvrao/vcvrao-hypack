/***********************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 1.3 : javathread_program_vectorvector_blockstrip.java
 
  Objective   : To Compute vector vector multiplication using block striped method
                and Demostrate use of 
			Thread().
			start().
			run().
			Join().
			Runnable interface.

  Input       : Vector size,no of thread,no of thread must be factor of vector size.

  Output      : Dot product of vector.
  
  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************************/

import java.util.*;
public class javathread_program_vectorvector_blockstrip implements Runnable
{

	 /* Delare variable */

	 int vSize,numThread;
	 double sum = 0.0;
	 int dist =0;
	 long startTime,endTime,diffTime;
	 double [] vecA; 
	 double [] vecB;
	 Thread [] t;
         /* main function start */

	 public static void main(String args[])
	{

                int vSize1,numThread1;
                /* check for correct number of argument */

		if(args.length != 2)
		{
			System.out.println(" Invalid Number of Argument It Must Be Two ");
			return ;
		}
		 
		vSize1 = Integer.parseInt(args[0]);
		numThread1 = Integer.parseInt(args[1]);
		/*check no of thread not more then 8 */

		if(numThread1 > 8)
		{
			System.out.println("no of thread could not be more then 8 ");
			 return ;
		}
		
		if(vSize1 % numThread1 != 0)
		{
			System.out.println("vector size is not a factor of number of thread ");
			 return ;
		}

		javathread_program_vectorvector_blockstrip v = new javathread_program_vectorvector_blockstrip();
		v.read_vector(vSize1,numThread1);
		v.serial_vector_multi(vSize1);
	}
	
	/* read function start here */

	public void read_vector(int vSize,int numThread)
	{
		
		int counter;
		this.vSize = vSize;
		this.numThread = numThread;
		Random rn = new Random();
		vecA = new double[vSize];
		vecB = new double[vSize];

	        /*initialization of  vector A and vector B */

		for(counter = 0; counter < vSize ; ++counter)
		{
			vecA[counter] = rn.nextDouble();
			vecB[counter] = rn.nextDouble();
		}

		dist = vSize / numThread;

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
		diffTime = endTime - startTime;
		System.out.printf(" Sum is %5.3f \n",sum);		
		System.out.println(" Total Time in Millisecond is  " + diffTime );
	}

        /* function for serial vector multiplication */
   
	public void  serial_vector_multi(int vSize)
	{
	//	this.vSize = vSize;
		double cSum =0;
		int counter;
		long  startTime1,endTime1,diffTime1;
		startTime1 = System.currentTimeMillis();
		for(counter = 0; counter < vSize ; ++counter)
		{
			cSum +=  vecA[counter] * vecB[counter];
		}
		
		endTime1 = System.currentTimeMillis();
		diffTime1 = endTime1 - startTime1;
		System.out.printf(" serial  Sum is %5.3f \n",cSum);		
		System.out.println(" Total Time for  serial  Multiplication in Millisecond  " + diffTime1);		

	}

	
       /* create thread function start here */
	
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
/* call back function run start here */
	
	public void run()
	{
		Thread tt = Thread.currentThread();

		int no=Integer.parseInt(tt.getName()) + 1;
		double localSum=0.0;
		int counter;
				
		System.out.println( no + " : i am taking interval " + ((no-1) * dist) + " from " + ((no * dist) -1));  
	/* logic to calculate local sum */
		for(counter =((no -1) * dist); counter <= ((no * dist) -1) ; ++counter)
		{
			localSum += vecA[counter] * vecB[counter];
		}

            /* synchronized block to compute global sum */

		synchronized(this)
		{
			sum +=  localSum;
		}
	}

        


}


