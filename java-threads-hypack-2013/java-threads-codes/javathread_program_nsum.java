/****************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 1.1 : javathread_program_nsum.java
 
  Objective   : To compute the sum of n number using number that assign to thread 
                and Demostrate use of 
			Runnable interface.
			Thread().
			run().

  Input       : Number of thread.

  Output      : Sum of n natural number.

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************************/

class NewThreadDemo implements Runnable
{
	Thread t;
	int tNum;
	static int sum=0;
	String tName;

/*	static class theLock extends Object {
   	}
  	 static public theLock lockObject = new theLock(); */

	NewThreadDemo(String no)
	{
		tNum =Integer.parseInt(no);

		t = new Thread(this,no);

		t.start();
	}	
	
	public void run()
	{
		try
		{
			System.out.println("i am  Thread " + tNum );
                        
			synchronized(this)
			{

				sum +=tNum;
			}
		}
		catch(Exception e)
		{
		  	System.out.println(" thread intrupted ");
		}
		
		System.out.println("Thread " + tNum +" :Exit");

	}

} 

public class javathread_program_nsum
{	
	public static void main(String args[])
	{
		
		int NumOfThread,i;
       		if(args.length == 0)
		{
			System.out.println("invalid number of argument ");
			return;
		}

		NumOfThread = Integer.parseInt(args[0]);	
		
		if(NumOfThread >8)
		{
			System.out.println("no of thread could not be more then 8 ");
			return;
		}	
		
		if(NumOfThread == 0)
		{
			System.out.println("we assume number of thread five ");
			NumOfThread = 5;
		}	
		
		NewThreadDemo t[]=new NewThreadDemo[NumOfThread];
		for(i = 1;i<=NumOfThread; ++i)
		{
			try
			{
				t[i-1]=new NewThreadDemo(""+i);
				t[i-1].t.join();
			}
			catch(InterruptedException e)
			{
				System.out.println(" Main Thread Intrupted ");
			}

		}


		System.out.println("sum is "+NewThreadDemo.sum);
		System.out.println(" Main thread Exit ");
	}
}







