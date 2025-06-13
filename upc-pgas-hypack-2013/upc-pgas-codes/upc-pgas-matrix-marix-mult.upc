/**************************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013
	
	(UPC)
	Matrix-Matrix multiplication using UPC (MAT1 X MAT2)
	Here MAT1 is distributed among all the threads row wise
	MAT2 is globally shared by all the threads

	ROWS1 X COLS1 are dimensions of Matrix1
	ROWS2 X COLS2 are dimensions of Matrix2

	Matrix dimensions can be changed by the user such that, 
        COLS1=ROWS2 and ROWS1,COLS2 must be multiple of THREADS.

Compiling:
	upcc -network=mpi -uses-mpi -link-with=mpicc mat_mul.upc 
            -I/usr/local/mpich2-1.0.7/include/ -lmpich 

Execution:
		upcrun -n 8 ./a.out

Result:
		Execution time in milliseconds.

   Created      : August-2013

   E-mail       : hpctfe@cdac.in     

***********************************************************************/


#include<upc.h>
#include<stdio.h>
#include<math.h>
#include<fcntl.h>
#define ROWS1 (50*THREADS)
#define COLS1 (400)   //must be equal to ROWS2
#define ROWS2 (400)   //must be equal to COLS1
#define COLS2 (50*THREADS)
#define res ROWS1*COLS2
shared [COLS1] double MAT1[ROWS1*COLS1];//Give a row to each thread
int main()
{
        shared double *MAT2,*RES;
        MAT2=(shared double*)upc_all_alloc(ROWS2*COLS2,sizeof(double));//MAT2 is shared by all the threads
        RES=(shared double*)upc_all_alloc(ROWS1*COLS2,sizeof(double));


  	/* generate random values for MAT1 */
  	for(int j=0;j<COLS1;j++)
        {
        	upc_forall(int i=0;i<ROWS1;i++;&MAT1[i*COLS1+j])
        		MAT1[i*COLS1+j]=(double)rand()/25.0;
        } 


  	/* generate random values for MAT2 */
        if(MYTHREAD==0)
  	for(int i=0;i<ROWS2;i++)
  	{
  		for(int j=0;j<COLS2;j++)
        	{
 			MAT2[i*COLS2+j]=(double)rand()/25.0;
		}
  	}
        
	/* initialise result matrix */
        upc_memset(RES,0,ROWS1*COLS2);

        upc_barrier 100;  

        
  	/*...................matrix multiplication starts here.............................*/


        if(MYTHREAD==0)printf("MAT-MAT MUltiplication Started....\n");
        struct timeval t1,t2;
        gettimeofday(&t1,NULL);

		/* Matrix multiplication */
  		for(int j=MYTHREAD,k=0;k<COLS2;j=(j+1)%COLS2,k++)//select a column in MAT2
                {
                	for(int l=0;l<COLS1;l++)
                        {
                        	upc_forall(int i=0;i<ROWS1;i++;&MAT1[i*COLS1+l])//select a row from MAT1
                               		RES[i*COLS2+j]+=MAT1[i*COLS1+l]*MAT2[l*COLS2+j];
			}
                        upc_barrier;//exchange columns in MAT2
                }


        upc_barrier 10001;
        gettimeofday(&t2,NULL);
        if(MYTHREAD==0)
        {
		int time_taken=(t2.tv_sec-t1.tv_sec)*1000000+t2.tv_usec-t1.tv_usec;
                printf("Time taken for mat-mat mul is %d microseconds for matrix size %d x %d x %d\n",time_taken,ROWS1,COLS1,COLS2);
        }
        upc_barrier 101; 
        upc_global_exit(0);
}
