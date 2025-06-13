/****************************************************************************                
  Objective     : Matrix - Matrix Multiplication using

                  OpenMP one PARALLEL for directive and Private Clause

                  OpenMP three PARALLEL for directive and Private Clause

                  OpenMP one PARALLEL for directive with Schedule(static) and Private Clause
                 
   Input        : Size of Matrices(i.e Size of Matrix A and Matrix B)

                  Number of Threads

   Output       : Number of Threads

                  Total Memory Utilized for the Matrix - Matrix Computation

                  Total Time Taken for Matrix - Matrix Computaion 

******************************************************************************/

#include <stdio.h>
#include <sys/time.h>
#include <omp.h>

#include"input_paramaters.h"
#if CLASS == 'A'
#define  CLASS_SIZE  1024
#endif

#if CLASS == 'B'
#define  CLASS_SIZE  2048
#endif

#if CLASS == 'C'
#define  CLASS_SIZE  4096
#endif

/*              Function declaration            */ 

double Matrix_Multiplication_One(int **Matrix_A,int **Matrix_B,int **Result,int N_size,int Total_threads);

double Matrix_Multiplication_Three(int **Matrix_A,int **Matrix_B,int **Result,int N_size,int Total_threads);

double Matrix_Multiplication_Static(int **Matrix_A,int **Matrix_B,int **Result,int N_size,int Total_threads);

int **ResultMat,count=0;
/* Main Program */

main()
{
	int             N_size, i,j,k,Total_threads;
	double          Total_overhead = 0.0;
	int   		**Matrix_A, **Matrix_B, **Result;
        double          memoryused = 0.0;
        int             iteration;
    
        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Centre for Development of Advanced Computing (C-DAC):  December-2006");
        printf("\n\t\t C-DAC Multi Core Benchmark Suite 1.0");
        printf("\n\t\t Email : betatest@cdac.in");
        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Objective : Dense Matrix Computations (Integer Operations)\n ");
        printf("\n\t\t Performance of three different Matrix Computation suites");
        printf("\n\t\t Matrix into Matrix Multiplication using ");
        printf("\n\t\t OpenMP one PARALLEL for directive and Private Clause;");
        printf("\n\t\t OpenMP three PARALLEL for directive and Private Clause;");
        printf("\n\t\t OpenMP one PARALLEL for directive with Schedule(static) and Private Clause");
        printf("\n\t\t..........................................................................\n");

        N_size = CLASS_SIZE;
        Total_threads = THREADS;  
        printf("\n\t\t Matrix Size :  %d",N_size);
        printf("\n\t\t Threads     :  %d",Total_threads);
        printf("\n");

	
	/* Matrix_A Elements */

	Matrix_A = (int **) malloc(sizeof(int *) * N_size);
	for (i = 0; i < N_size; i++) {
		Matrix_A[i] = (int *) malloc(sizeof(int) * N_size);
		for (j = 0; j < N_size; j++)
		{
			Matrix_A[i][j] = i+j;
		}
	}

	/* Matrix_B Elements */

	Matrix_B = (int **) malloc(sizeof(int *)  * N_size);
	for (i = 0; i < N_size; i++) {
		Matrix_B[i] = (int *) malloc(sizeof(int) * N_size);
		for (j = 0; j < N_size; j++)
		{
                        Matrix_B[i][j] = i+j;
		}
	
	}

	ResultMat = (int **) malloc(sizeof(int *) * N_size);
	for (i = 0; i < N_size; i++) 
	ResultMat[i] = (int *) malloc(sizeof(int) * N_size);

        for(i=0;i<N_size;i++)
        for(j=0;j<N_size;j++)
        ResultMat[i][j] =0;
        for(k=0;k<N_size;k++)
        {
         for(i=0;i<N_size;i++)
          {
           for(j=0;j<N_size;j++)
           ResultMat[k][i] +=Matrix_A [k][j] * Matrix_B[j][i];
          }
        }

	/* Dynamic Memory Allocation */

	Result = (int **) malloc(sizeof(int *) * N_size);
	for (i = 0; i < N_size; i++) 
		Result[i] = (int *) malloc(sizeof(int) * N_size);
	
        memoryused = (4*(N_size*N_size))*sizeof(int *);


	/* Function Calling   */
	
       Total_overhead = Matrix_Multiplication_One(Matrix_A,Matrix_B,Result,N_size,Total_threads);
       
       Total_overhead += Matrix_Multiplication_Three(Matrix_A,Matrix_B,Result,N_size,Total_threads);

       Total_overhead += Matrix_Multiplication_Static(Matrix_A,Matrix_B,Result,N_size,Total_threads);

       
       printf("\n\t\t Memory Utilized         : %lf MB ",(memoryused/(1024*1024)));
       printf("\n\t\t Time in Seconds (T)     : %.1lf ",Total_overhead);	
       printf("\n\t\t   ( T represents the Time taken to execute the three suites )");
       printf("\n\t\t..........................................................................\n");
	
	/* Free Memory     */
	free(Matrix_A);
	free(Matrix_B);
	free(Result);
        free(ResultMat);

        print_info( "Matrix Matrix Computation ", CLASS, THREADS,Total_overhead,
               COMPILETIME, CC, CLINK, C_LIB, C_INC, CFLAGS, CLINKFLAGS );
      
 
}/* Main function end    */

/* Functions implementation   */

double Matrix_Multiplication_One(int **Matrix_A,int **Matrix_B,int **Result,int N_size,int Total_threads)
{
	int             i,j,k;
	struct timeval  TimeValue_Start;
	struct timezone TimeZone_Start;

	struct timeval  TimeValue_Final;
	struct timezone TimeZone_Final;
	double            time_start, time_end;
	double            time_overhead;
	
	gettimeofday(&TimeValue_Start, &TimeZone_Start);

	/* OpenMP Parallel For Directive */

	omp_set_num_threads(Total_threads);
	#pragma omp parallel for private(j,k)
	for (i = 0; i < N_size; i = i + 1)
		for (j = 0; j < N_size; j = j + 1){
			Result[i][j]=0;
			for (k = 0; k < N_size; k = k + 1){
				Result[i][j] = Result[i][j] + Matrix_A[i][k] * Matrix_B[k][j];
                        }
		}
          
	for (i = 0; i < N_size; i ++)
        {
           for (j = 0; j < N_size; j ++)
           {
             if(Result[i][j] != ResultMat[i][j])
             count++;
           }
         }
	gettimeofday(&TimeValue_Final, &TimeZone_Final);
      

	time_start =(double) TimeValue_Start.tv_sec * 1000000 +(double) TimeValue_Start.tv_usec;
	time_end = (double)TimeValue_Final.tv_sec * 1000000 + (double)TimeValue_Final.tv_usec;
	time_overhead = (time_end - time_start)/1000000.0;

	printf("\n\t\t Matrix into Matrix Multiplication using one Parallel for pragma......Done \n");
        if(count == 0)
	printf("\n\t\t Matrix into Matrix Multiplication using one Parallel for pragma......Successful \n");

        count=0;	
	return time_overhead;
}

double Matrix_Multiplication_Three(int **Matrix_A,int **Matrix_B,int **Result,int N_size,int Total_threads)
{
	int             i,j,k;
	struct timeval  TimeValue_Start;
	struct timezone TimeZone_Start;

	struct timeval  TimeValue_Final;
	struct timezone TimeZone_Final;
	double            time_start, time_end;
	double            time_overhead;
	
	gettimeofday(&TimeValue_Start, &TimeZone_Start);
	

	/* OpenMP Three For Directive    */ 

	#pragma omp parallel for private (j,k) shared (Matrix_A,Matrix_B,Result,N_size) num_threads(Total_threads) 
	for (i = 0; i < N_size; i = i + 1){
        #pragma omp parallel for private(k) shared (Matrix_A,Matrix_B,Result,N_size) num_threads(Total_threads) 	
	for (j = 0; j < N_size; j = j + 1){
		  Result[i][j]=0;
                  #pragma omp parallel for private(k) shared (Matrix_A,Matrix_B,Result,N_size) num_threads(Total_threads)
			for (k = 0; k < N_size; k = k + 1)
				Result[i][j] = Result[i][j] + Matrix_A[i][k] * Matrix_B[k][j];

	}
      }
	for (i = 0; i < N_size; i ++)
        {
           for (j = 0; j < N_size; j ++)
           {
             if(Result[i][j] != ResultMat[i][j])
             count++;
           }
         }
	gettimeofday(&TimeValue_Final, &TimeZone_Final);

	time_start =(double) TimeValue_Start.tv_sec * 1000000 + (double)TimeValue_Start.tv_usec;
	time_end =(double) TimeValue_Final.tv_sec * 1000000 + (double)TimeValue_Final.tv_usec;
	time_overhead = (time_end - time_start)/1000000.0;

	printf("\n\t\t Matrix into Matrix Multiplication using three Parallel for pragma......Done \n");
        if(count == 0)
	printf("\n\t\t Matrix into Matrix Multiplication using three Parallel for pragma......Successful \n");

        count=0;	
	
	return time_overhead; 
}

double Matrix_Multiplication_Static(int **Matrix_A,int **Matrix_B,int **Result,int N_size,int Total_threads)
{
	int	i,j,k;
	struct timeval  TimeValue_Start;
	struct timezone TimeZone_Start;

	struct timeval  TimeValue_Final;
	struct timezone TimeZone_Final;
	double            time_start, time_end;
	double            time_overhead;
	
	gettimeofday(&TimeValue_Start, &TimeZone_Start);
	
	/* OpenMP For Directive with static option    */	
	
	#pragma omp parallel for private(j,k) schedule(static)
	for (i = 0; i < N_size; i = i + 1){
		for (j = 0; j < N_size; j = j + 1){
			Result[i][j]=0;
			for (k = 0; k < N_size; k = k + 1)
				Result[i][j] = Result[i][j] + Matrix_A[i][k] * Matrix_B[k][j];
		}
	}

	for (i = 0; i < N_size; i ++)
        {
           for (j = 0; j < N_size; j ++)
           {
             if(Result[i][j] != ResultMat[i][j])
             count++;
           }
         }
	 gettimeofday(&TimeValue_Final, &TimeZone_Final);

	time_start =(double) TimeValue_Start.tv_sec * 1000000 +(double) TimeValue_Start.tv_usec;
	time_end = (double)TimeValue_Final.tv_sec * 1000000 + (double)TimeValue_Final.tv_usec;
	time_overhead = (time_end - time_start)/1000000.0;

	printf("\n\t\t Matrix into Matrix Multiplication using one Parallel for pragma with static option......Done \n");
        if(count == 0)
	printf("\n\t\t Matrix into Matrix Multiplication using one Parallel for pragma with static option......Successful \n");

        count=0;	

	return time_overhead;
}







