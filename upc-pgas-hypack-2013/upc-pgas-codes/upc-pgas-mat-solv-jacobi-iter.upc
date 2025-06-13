/******************************************************************************** 

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

	Jacobis method for strictly diagonally dominant matrix using UPC
	"size" is  size of the  Matrix size u want to solve,
        this can be changed by the user and should be a positive integer.

	EPSILON is tolerance value, can be changed by the user, should be a positive value.
	MAX_ITERATIONS limits the number of iterations ,it can be changed by user.

Compiling:
		upcc jacobi.upc	
Executing:
		upcrun -n 8 ./a.out

Created         : August-2013

E-mail          : hpctfe@cdac.in     

*************************************************************************************/

#include<stdio.h>
#include<upc.h>
#include<math.h>
#include<sys/time.h>
#include<unistd.h>
#include<fcntl.h>
#define size (200)		//Can be Changed by the user
#define SIZE (size+1) 		//Should Not be changed
#define ABS(a) ((a)<0)?(-a):(a)
#define EPSILON 1E-09
#define MAX_ITERATIONS (1000*50)
int main()
{
	struct timeval t1,t2;
	int i,j,BAR_VAL=0;int NOOF_ITERATIONS=0;
	int present=1;//indicates which Array we are using to store result OLD_X or NEW_X
	shared double* Matrix;
	Matrix=(shared double*)upc_all_alloc(SIZE*(SIZE-1),sizeof(double));

 
	/* Generate a diagonally dominant matrix having solution as all 1's */

   	if(MYTHREAD==0)
    	{
                printf("Enter any integer to Start :");
                int jk;scanf("%d",&jk);
       
		//Initialisie Matrix(SIZE,SIZE-1) to ZEROS
		for(int i=0;i<SIZE-1;i++)
			for(int j=0;j<SIZE-1;j++)
				Matrix[i*(SIZE-1)+j]=0.0;



		//Fill the matrix with values and satisfy the diagonal dominance
      		for(i=0;i<SIZE*(SIZE-1);i++)
        	{ 
                        Matrix[i]=(double)i;
        	}



		/* Make Diagonally dominant by Setting Diagonal elements */
		double k=(SIZE-1)*SIZE/2.0;
                for(i=0;i<SIZE-1;i++)
                {
			Matrix[i*(SIZE)+i]=((i*SIZE)-1)*(SIZE-1)+k;	
                }


		/* set the Matrix B in AX=B */
		for(int i=0;i<SIZE-1;i++)
		{	
			Matrix[(i+1)*SIZE-1]=0.0;
			for(int j=0;j<SIZE-1;j++)
			{
				Matrix[(i+1)*SIZE-1]+=Matrix[(i)*SIZE+j];
			}
		}



		/* Print the Diagonal Dominant input Matrix */
		/*printf("%d %d \n",SIZE-1,SIZE-1);
                for(i=0;i<SIZE*(SIZE-1);i++)
                {	
                        if(i%(SIZE)==0&&i!=0)printf("\n");
                        if((i+1)%SIZE!=0) printf("%lf ",Matrix[i]);
                }*/
 		
           
	       printf("\nCompleted generating random input matrix..\nSolving the matrix...\n");
     		
    	}

 	upc_barrier;
	gettimeofday(&t1,NULL);


        //divide each row by corresponding  diagonal element
        int k;
        upc_forall(i=0;i<SIZE-1;i++;i%THREADS)//for all rows
        {
		for(j=0,k=Matrix[i*SIZE+i];j<SIZE-1;j++)//for that whole row
                {
 			Matrix[i*SIZE+j]/=-k;
                }
                Matrix[i*SIZE+j]/=k;
                Matrix[i*SIZE+i]=0.0;
        }

        
     	upc_barrier BAR_VAL;BAR_VAL++;


	/* Allocate Shared memory for storing results....*/
    	shared double *OLD_X,*NEW_X;
    	OLD_X=(shared double*)upc_all_alloc(SIZE-1,sizeof(double));
    	NEW_X=(shared double*)upc_all_alloc(SIZE-1,sizeof(double));
    

	if(OLD_X==NULL||NEW_X==NULL)
      	{
        	printf("Error in allocating partitioned memory...");
        	upc_global_exit(-1);
      	} 
    	upc_barrier BAR_VAL;BAR_VAL++;
    
    	shared int* compare;
    	compare=(shared int*)upc_all_alloc(1,sizeof(int));
    	upc_memset(OLD_X,0,SIZE-1);



	do{

		//if(MYTHREAD==0)printf("%d ",NOOF_ITERATIONS);
                NOOF_ITERATIONS++;
       	 	/*printf("This is Thread %d.....\n ",MYTHREAD);   */
           	k=MYTHREAD*SIZE;
        	upc_barrier BAR_VAL;BAR_VAL++;
          	*compare=1;
        	while(k<(SIZE*(SIZE-1)))
        	{
          		if(present)//use old
          		{
            			NEW_X[k/SIZE]=0.0;
            			for(i=k,j=0;(i<k+SIZE)&&(j<SIZE-1);i++,j++)
             			{
               				NEW_X[k/SIZE]+=Matrix[i]*OLD_X[j];
             			}   
             			NEW_X[k/SIZE]+=Matrix[i];
             			if(fabs(NEW_X[k/SIZE]-OLD_X[k/SIZE])>EPSILON)*compare=0;
          		}
           
          		else
          		{
            			OLD_X[k/SIZE]=0.0;
            			for(i=k,j=0;(i<k+SIZE)&&(j<SIZE-1);i++,j++)
             			{
               				OLD_X[k/SIZE]+=Matrix[i]*NEW_X[j];
             			}
             			OLD_X[k/SIZE]+=Matrix[i];
             			if(fabs(OLD_X[k/SIZE]-NEW_X[k/SIZE])>EPSILON)*compare=0;
          		}
 		        k+=SIZE*THREADS;
		}
        	upc_barrier BAR_VAL;BAR_VAL++;//
         	present=(present+1)%2;
	      

              /* Print solution obtained after each iteration */            	
             /* if(MYTHREAD==0)
             	{
               		for(i=0;i<SIZE-1;i++)
                 		printf("%lf ",OLD_X[i]);
                 		printf("\n");
               		for(i=0;i<SIZE-1;i++)
                 		printf("%lf ",NEW_X[i]);
                 		printf("\n");
             	}*/
	       

	}while(*compare==0&&MAX_ITERATIONS>NOOF_ITERATIONS);


	/* Print the Final Solution */
	if(MYTHREAD==0)
                {
			printf("Result Matrix is : \n");
			gettimeofday(&t2,NULL);
			int time=(t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec;
			/* Solution in OLD Matrix */
                        for(i=0;i<SIZE-1;i++)
                                printf("%lf ",OLD_X[i]);
                                printf("\n");
			/* Solution in NEW Matrix */
                        for(i=0;i<SIZE-1;i++)
                                printf("%lf ",NEW_X[i]);
                                printf("\n");
			 printf("\nNo of Iterations=%d.\nTIme taken=%d microseconds.\nInput Matrix size=%d.\nTolerance=%e.\n",NOOF_ITERATIONS,time,size,EPSILON);
                }


	/* Free the allocated Memories */
	upc_free(OLD_X);
      	upc_free(NEW_X);
      	upc_free(Matrix);
      	upc_free(compare);
      	upc_global_exit(0);
}

