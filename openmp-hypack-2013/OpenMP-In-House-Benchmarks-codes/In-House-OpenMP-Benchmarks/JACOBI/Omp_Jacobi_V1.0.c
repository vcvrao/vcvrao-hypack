/*************************************************************************
        Objective           :  Jacobi method to solve AX = b matrix  system
                              of linear equations using OpenMP Parallel For
                              Directive. 

        Input               : Number Of Threads &
                              Real Symmetric Positive definite Matrix size
                              Read from Jacobi_Data.inp

        Description         : Input matrix is stored in n by n format.
                             Diagonal preconditioning matrix is used.Rowwise
                             block striped partitioning matrix is used.
                             Maximum iterations is given by MAX_ITERATIONS
                             Tolerance value is given by EPSILON


        Output              : The solution of  Ax=b and the number of
                             iterations for convergence of the method.

       Necessary conditions : Input Matrix Should be Square Matrix.

**************************************************************************/

#include <stdio.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>  
#define  MAX_ITERATIONS 10000

/* // commented by shiva 
 
#define CLASS_A 2048 
#define CLASS_B 4096  
#define CLASS_C 8192
*/
/* Added by shiva dec28 */
/* start */
#include"input_paramaters.h"
#if CLASS == 'A'
#define  CLASS_SIZE  2048
#endif

#if CLASS == 'B'
#define  CLASS_SIZE  4096
#endif

#if CLASS == 'C'
#define  CLASS_SIZE  8192
#endif
/*end */



double  Distance(double *X_Old, double *X_New, int n_size);

main(int argc, char **argv)
{

	/* .......Variables Initialisation ...... */
        double    diag_dominant_factor  = 2.0000;
        double    tolerance  = 1.0E-15;
	int       n_size, NoofRows_Bloc, NoofRows, NoofCols;
	int       Numprocs, MyRank, NumThreads, Root = 0;
	int       irow, jrow, icol, index, Iteration, GlobalRowNo;
	int       Threadid, Noofthreads,iteration;
	
	double    **Matrix_A, *Input_A, *Input_B, *ARecv, *BRecv;
	double    *X_New, *X_Old, *Bloc_X, tmp,rno,sum;
        struct    timeval  TimeValue_Start;
        struct    timezone TimeZone_Start;

        struct    timeval  TimeValue_Final;
        struct    timezone TimeZone_Final;
        double    time_start, time_end;
        double    time_overhead;
        double     memoryused = 0.0;
	/* // Coomented by shiva 
        FILE       *fp;*/


	/* Checking For Square Matrix Input File */
        /* // commented by shiva 
        fp = fopen("./data/Jacobi_Data.inp", "r");
        if (!fp)
        {
           printf("\nUnable to open the file Jacobi_Data.dat");
           exit(0);
        }
 	fscanf(fp, "%d", &NumThreads);
        fclose(fp);
        */
 
        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Centre for Development of Advanced Computing (C-DAC):  December-2006");
        printf("\n\t\t C-DAC Multi Core Benchmark Suite 1.0");
        printf("\n\t\t Email : betatest@cdac.in");
        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Objective : To Solve AX=B Linear Equation (Jacobi Method)\n ");
        printf("\n\t\t Performance for solving AX=B Linear Equation using JACOBI METHOD");
        /* //commented by shiva 
        printf("\n\t\t on Multi Socket Multi Core Processor using 1/2/4/8 threads \n");
        printf("\n\t\t Input Parameters :");
        printf("\n\t\t CLASS A - Square Matrix Size : 2048");
        printf("\n\t\t CLASS B - Square Matrix Size : 4096");
        printf("\n\t\t CLASS C - Square Matrix Size : 8192");
        printf("\n\t\t..........................................................................\n");
       
        for(iteration = 0;iteration < 3;iteration++)
        {
         if(iteration == 0)
         {
          n_size = CLASS_A;
          printf("\n\t\t CLASS A ");
         }
         if(iteration == 1)
         {
          n_size = CLASS_B;
          printf("\n\t\t CLASS B ");
         }
         if(iteration == 2)
         {
          n_size = CLASS_C;
          printf("\n\t\t CLASS C ");
         }*/
         n_size = CLASS_SIZE;
         NumThreads = THREADS; 
         printf("\n\t\t Matrix Size :  %d",n_size);
         printf("\n\t\t Threads     :  %d",NumThreads);


	NoofRows = n_size; 
	NoofCols = n_size; 


	/* Necessary Condition */
        /* // Commented by shiva dec 28 
        if((NumThreads != 1) && (NumThreads != 2) && (NumThreads != 4) && (NumThreads  != 8))
        {
          printf("\n User Should give Only 1, 2 , 4 , 8 Number of Threads\n");
          exit(0);
        }

        // When the NoofRows=NoofCols=n_size ..whatz need of condition..!
	if (NoofRows != NoofCols) {
		printf("Input Matrix Should Be Square Matrix ..... \n");
		exit(-1);
	}
        */

	/* ...Allocate Memory And Read Data ..... */

	Matrix_A = (double **) malloc(n_size * sizeof(double *));
	Input_B = (double *) malloc(n_size * sizeof(double));
        memoryused += ( n_size *sizeof(double));
	for (irow = 0; irow < n_size; irow++) {
		Matrix_A[irow] = (double *) malloc(n_size * sizeof(double));
		sum=0.0;
		for (icol = 0; icol < n_size; icol++){
			srand48((unsigned int)NoofRows);
                         
                        rno = (double)(rand()%10)+1.0;
                        if(irow == icol ) rno = 0.0;
                        Matrix_A[irow][icol]= rno;
			sum+=Matrix_A[irow][icol];
		}
			Matrix_A[irow][irow]= sum * diag_dominant_factor;
		Input_B[irow] = sum + Matrix_A[irow][irow];
		
	}

        memoryused += ( n_size * n_size * sizeof(double));


	/* Dynamic Memory Allocation */

	X_New = (double *) malloc(n_size * sizeof(double));
        memoryused += ( n_size * sizeof(double));

	X_Old = (double *) malloc(n_size * sizeof(double));
        memoryused += ( n_size * sizeof(double));

	Bloc_X = (double *) malloc(n_size * sizeof(double));
        memoryused += ( n_size * sizeof(double));

	/* Initailize X[i] = B[i] */

	for (irow = 0; irow < n_size; irow++)
        {
		Bloc_X[irow] = Input_B[irow];
		X_New[irow] = Input_B[irow];
        }

	//printf(" --------------------------------------------------- \n");
/*	printf("Initial Soluiuton X_New [ ] size of the matrix = %d\n", n_size);
	printf("\n");
	for (index = 0; index < n_size; index++) printf("%.3lf  ", X_New[index]);
  */     
	//printf(" ---------------Initial Solution DONE  \n");
        gettimeofday(&TimeValue_Start, &TimeZone_Start);
 
	Iteration = 0;
	do {

	      for(index = 0; index < n_size; index++) X_Old[index] = X_New[index];

	
               #pragma omp parallel for private(icol) shared(Bloc_X,X_New,X_Old)

		for (irow = 0; irow < n_size; irow++) 
		{
			Threadid = omp_get_thread_num();
/*			printf("\n Jacobi : thread-id %d\n", Threadid);   */ 
			Noofthreads = omp_get_num_threads();
/*			printf(" Jacobi Total number of threads for this execution are %d\n", Noofthreads);   */

			Bloc_X[irow] = Input_B[irow];

			for (icol = 0; icol < irow; icol++) {
				Bloc_X[irow] -= X_Old[icol] * Matrix_A[irow][icol];
			}
			for (icol = irow + 1; icol < n_size; icol++) {
				Bloc_X[irow] -= X_Old[icol] * Matrix_A[irow][icol];
			}
			Bloc_X[irow] = Bloc_X[irow] / Matrix_A[irow][irow];

//		 printf("\n Jacobi : thread-id %d  Iteration - %d \n", Threadid, Iteration);  

		}

	         for(index = 0; index < n_size; index++) X_New[index] = Bloc_X[index];

	       Iteration++;
/*	for (irow = 0; irow < n_size; irow++)
		printf("%.3lf  ", X_New[irow]);
*/
	 } while ((Iteration < MAX_ITERATIONS) && (Distance(X_Old, X_New, n_size) >= tolerance));

        gettimeofday(&TimeValue_Final, &TimeZone_Final);

        time_start =(double) TimeValue_Start.tv_sec * 1000000 +(double) TimeValue_Start.tv_usec;
        time_end = (double)TimeValue_Final.tv_sec * 1000000 + (double)TimeValue_Final.tv_usec;
        time_overhead = (time_end - time_start)/1000000.0;



	/* .......Output Vector ..... */

/*
	printf("\n");
	printf("Results Of Jacobi Method \n");
	printf("\n");

	printf("Matrix Input_A \n");
	printf("\n");
	for (irow = 0; irow < n_size; irow++) {
		for (icol = 0; icol < n_size; icol++)
			printf("%.1lf  ", Matrix_A[irow][icol]);
		printf("\n");
	}

	printf("\n");
	printf("Matrix Input_B \n");
	printf("\n");
	for (irow = 0; irow < n_size; irow++) {
		printf("%.1lf  ", Input_B[irow]);
	}
	printf("\n");
	printf("Solution Vector \n");
        */
        printf("\n\t\t The Jacobi Method For AX=B .........DONE");
	printf("\n\t\t Total Number of Iterations : %d", Iteration);
        printf("\n\t\t Memory Utilised            : %lf MB \n",(memoryused/(1024*1024)));
        printf("\n\t\t Time in Seconds (T)        : %lf",time_overhead);
        printf("\n\t\t   ( T represents the Time taken to solve the Linear system of equation  )");
        printf("\n\t\t..........................................................................\n");
	/*
        for (irow = 0; irow < n_size; irow++)
		printf("%.3lf  ", X_New[irow]);*/
	//printf(" --------------------------------------------------- \n");
	printf("\n");
	/* Freeing Allocated Memory */

	free(X_New);
	free(X_Old);
	free(Matrix_A);
	free(Input_B);
	free(Bloc_X);
      /* // Commented by shiva dec 28 
      } */
   /* Added by shiva dec 28*/
 
  print_info( "Jacobi Computation ", CLASS, THREADS,time_end-time_start,
               COMPILETIME, CC, CLINK, C_LIB, C_INC, CFLAGS, CLINKFLAGS );

 
}


double 
Distance(double *X_Old, double *X_New, int n_size)
{
	int             index;
	double          Sum;

	Sum = 0.0;
	for (index = 0; index < n_size; index++)
		Sum += (X_New[index] - X_Old[index]) * (X_New[index] - X_Old[index]);

	return (Sum);
}

