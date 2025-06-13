/***********************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Example 1.4 : javathread_program_matrixmatrix_time_double.java
 
  Objective   : To Compute matrix matrix multiplication 
   		and Demostrate	

			Thread().
			start().
			run().
			Join().
			Runnable interface.

  Input       : Row,coloum for matrix a and b ,no of thread,no of thread must be factor of 
		matrix a row size.

  Output      : Resultant matrix.

  Created     : August-2013

  E-mail      : hpcfte@cdac.in     

****************************************************************************/

import java.util.*;
public class javathread_program_matrixmatrix_time_double implements Runnable
{

	 /*  Declare variable  */

         int row1,col1,row2,col2,numThread;
	 int sum,currentRow=0,rowDist=0;
	 double [][] matA;                              
	 double [][] matB;
	 double [][] matRes;	
	 double [][] matResS;	
	 Thread []t;
	 long startTime;
	 long endTime;
	 long diffTime=0;
	 public static void main(String args[])
	{
	
		int row11,col11,row22,col22,numThread1;
		
		 if(args.length != 5)
                {
                        System.out.println(" Invalid Number of Argument You Must Pass 5 Argument ");
                        return ;
                }

		/* Reading value from command line */
                row11 = Integer.parseInt(args[0]);
                col11 = Integer.parseInt(args[1]);              
                row22 = Integer.parseInt(args[2]);
                col22 = Integer.parseInt(args[3]);
                numThread1 = Integer.parseInt(args[4]);
         
              /* check no of thread could not more then 8 */
                if(numThread1 > 8)
                {
                        System.out.println("Number Of Thread Can Not Greater Then 8 ");
                                return ;
                }

              /* check matix multiplication is possible or not */

                if(col11 != row22)
                {
                        System.out.println("Matrix Multiplication Is Not Possible ");
                         return ;
                }

                if(numThread1 > row22)
                {
                        System.out.println("Number Of Thread Can Not Greater Then No of Row Of second Matrix ");
                                return ;
                }

		javathread_program_matrixmatrix_time_double obj = new javathread_program_matrixmatrix_time_double();
                obj.read_matrix(row11,col11,row22,col22,numThread1);
        }
   
     /* function to initilize the matrix and calculate the time */

        public void read_matrix(int row1,int col1,int row2,int col2,int numThread)
	{
   	        this.row1=row1;
                this.col1=col1;
		this.row2=row2;
                this.col2=col2;
		this.numThread=numThread;
		int counter,i,j;
		Random rn = new Random();
		matA =  new double [row1][col1];
		matB =  new  double [row2][col2];           
		matRes= new double [row1][col2];
		matResS= new double [row1][col2];
		rowDist = row1 / numThread;       /* find in how many row each thread operate */

	        /*initilization of matrix A */

		for(i = 0; i < row1; ++i)
		{
			for(j=0; j< col1; ++j)
			{
				
				matA[i][j] = rn.nextFloat();   
			}
		}

		 /*initilization of matrix B */

		for(i = 0; i < row2; ++i)
		{
			for(j=0; j< col2; ++j)
			{
				matB[i][j] = rn.nextFloat();  

			}
		}


                 /* initilize the resultant matrix */

		for(i = 0; i < row1; ++i)
		{
			for(j=0; j<col2; ++j)
			{
				matRes[i][j] =  0.0;   
			} 	
		}

                /* initilize the serial resultant matrix */

		for(i = 0; i < row1; ++i)
                {
                        for(j=0; j<col2; ++j)
                        {
                                matResS[i][j] =  0.0;
                        }
                }

 		long startTime1,endTime1,diffTime1;
		
		/* serial matrix multiplication */
			       
		startTime1 = System.currentTimeMillis();
		for(i = 0; i < row1; ++i)

                {
                        for(j=0; j<col2; ++j)
                        {
                           for(int k=0;k< col1; ++k)
			   { 
			       matResS[i][j] += matA[i][k] * matB[k][j];
			   }
                        }
                }
            
		endTime1 = System.currentTimeMillis();
		diffTime1 = endTime1 - startTime1;
		System.out.println("Total time for serial multiplication in millisecond  " + diffTime1);
		/*calling function to create thread */
                        

			creat_thread();                                          
                    
		/* join the thread to main thread */
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

		

               /* checking for both result are same or not */

		for(i=0;i<row1;++i)
                {
			for(j=0;j<col2;++j)
			{
				if(matRes[i][j] != matResS[i][j])
				{

					System.out.println("matrix is not same");
						//break;
						return;
				}
			}

		}     
		
			diffTime = (endTime - startTime);
			System.out.println(" Total time in millisecond " + diffTime);
	}
	
       /* function to create the thread */

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
	
      /* call back run function start here */

	public void run()
	{
		Thread tt = Thread.currentThread();
		int no=Integer.parseInt(tt.getName()) + 1;
		int localRow,i,j,counter;	
                System.out.println("i m thread " + no);
		/*loop that iterate no of row in which each thread operate*/
	        for(counter =((no -1) * rowDist); counter <= ((no * rowDist) -1) ; ++counter)
                {
                       


		                      
                         	localRow = counter;
			/* Loop for matrix multiplication */

			for( i=0;i< col2; ++i)
			{
				for(j=0; j< col1 ; j++)
				{
					matRes[localRow][i] += matA[localRow][j] * matB[j][i];
				}			
			}
		}
          }

}


