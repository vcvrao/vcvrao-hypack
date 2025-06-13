/************************************************************************************

     Objective  :  Jacobi method to solve AX = b matrix system 
                   of linear equations. 

     Input      :  Number of Threads,

                   Size of Real Symmetric Positive definite Matrix. 

     Output     :  The solution of  Ax=b and the number of iterations
                   for convergence of the method.
 
*************************************************************************************/
#include <stdio.h>
#include<pthread.h>
#include<sys/time.h>
#define  MAX_ITERATIONS 100000



#include"input_paramaters.h"
#if CLASS == 'A'
#define  CLASS_SIZE  1024
#endif

#if CLASS == 'B'
#define  CLASS_SIZE  2048
#endif

#if CLASS == 'C'
#define  CLASS_SIZE 4096
#endif


double   Distance(double *X_Old, double *X_New, int matrix_size);
pthread_mutex_t  mutex1 = PTHREAD_MUTEX_INITIALIZER;

double   **Matrix_A, *Input_B;
double   *X_New, *X_Old, *Bloc_X, rno,sum;

int Number; 
void jacobi(int);

main(int argc, char **argv)
{

        double diag_dominant_factor  = 4.0000;
        double tolerance  = 1.0E-10;
        /* .......Variables Initialisation ...... */
        int matrix_size,  NoofRows, NoofCols;
        int NumThreads,ithread;
        int irow, icol, index, Iteration,iteration;
        double time_start, time_end,memoryused;
        struct timeval tv;
        struct timezone tz;
        FILE *fp;

        pthread_attr_t pta;
        pthread_t *threads;
     
        memoryused =0.0;


        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Centre for Development of Advanced Computing (C-DAC):  December-2006");
        printf("\n\t\t C-DAC Multi Core Benchmark Suite 1.0");
        printf("\n\t\t Email : betatest@cdac.in");
        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Objective : To Solve AX=B Linear Equation (Jacobi Method)\n ");
        printf("\n\t\t Performance for solving AX=B Linear Equation using JACOBI METHOD");
        printf("\n\t\t..........................................................................\n");


         matrix_size = CLASS_SIZE;
         NumThreads = THREADS;
         printf("\n\t\t Matrix Size :  %d",matrix_size);
         printf("\n\t\t Threads     :  %d",NumThreads);
          
        NoofRows = matrix_size; 
        NoofCols = matrix_size;
        
        if((NumThreads != 1) && (NumThreads != 2) && (NumThreads != 4) && (NumThreads  != 8))
        {
          printf("\n User Should give Only 1, 2 , 4 , 8 Number of Threads\n");
          exit(0);
        }             
        
       
        /* Allocate The Memory For Matrix_A and Input_B */
        Matrix_A = (double **) malloc(matrix_size * sizeof(double *));
        Input_B = (double *) malloc(matrix_size * sizeof(double));

        /* Populating the Matrix_A and Input_B */
        for (irow = 0; irow < matrix_size; irow++)
        {  
                Matrix_A[irow] = (double *) malloc(matrix_size * sizeof(double));
                sum=0.0;
                for (icol = 0; icol < matrix_size; icol++)
                {
                        srand48((unsigned int)NoofRows);

                        rno = (double)(rand()%10)+1.0;

                        if(irow == icol ) rno = 0.0;
                        Matrix_A[irow][icol]= rno;
                        sum+=Matrix_A[irow][icol];
                }
                Matrix_A[irow][irow]= sum * diag_dominant_factor;
                Input_B[irow] = sum + Matrix_A[irow][irow];
         }
        
         memoryused+=(NoofRows * NoofCols * sizeof(double));
         memoryused+=(NoofRows * sizeof(double));  
         
         printf("\n");

         if (NoofRows != NoofCols) 
         {
                printf("Input Matrix Should Be Square Matrix ..... \n");
                exit(-1);
         }

        /* Dynamic Memory Allocation */
        X_New = (double *) malloc(matrix_size * sizeof(double));
        memoryused+=(NoofRows * sizeof(double));
        X_Old = (double *) malloc(matrix_size * sizeof(double));
        memoryused+=(NoofRows * sizeof(double));
        Bloc_X = (double *) malloc(matrix_size * sizeof(double));
        memoryused+=(NoofRows * sizeof(double));

        /* Calculating the time of Operation Start */
        gettimeofday(&tv, &tz);
        time_start= (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

        /* Initailize X[i] = B[i] */

        for (irow = 0; irow < matrix_size; irow++)
        {
                Bloc_X[irow] = Input_B[irow];
                X_New[irow] =  Input_B[irow];
        }
  
        /* Allocating the memory for user specified number of threads */
        threads = (pthread_t *) malloc(sizeof(pthread_t) * NumThreads);  

        /* Initializating the thread attribute */
        pthread_attr_init(&pta);

        Iteration = 0;
        do {
              for(index = 0; index < matrix_size; index++) 
              X_Old[index] = X_New[index];
              for(ithread=0;ithread<NumThreads;ithread++)
                {

                 /* Creating The Threads */                 
                 pthread_create(&threads[ithread],&pta,(void *(*) (void *))jacobi, (void *) (matrix_size));

                }

                  Iteration++;
                  for (ithread=0; ithread<NumThreads; ithread++)
                  {
                  pthread_join(threads[ithread], NULL); 
                  }                  
                  pthread_attr_destroy(&pta);
             } while ((Iteration < MAX_ITERATIONS) && (Distance(X_Old, X_New, matrix_size) >= tolerance));
       
        /* Calculating the time at the end of Operation */
        gettimeofday(&tv, &tz);
        time_end= (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

        printf("\n\t\t The Jacobi Method For AX=B .........DONE"); 
        printf("\n\t\t Total Number Of Iterations   :  %d",Iteration-1);
        printf("\n\t\t Memory Utilized              :  %lf MB",(memoryused/(1024*1024)));
        printf("\n\t\t Time in  Seconds (T)         :  %lf",(time_end - time_start));
        printf("\n\t\t   ( T represents the Time taken to execute the four suites )");
        printf("\n\t\t..........................................................................\n");

        /* Freeing Allocated Memory */
        free(X_New);
        free(X_Old);
        free(Matrix_A);
        free(Input_B);
        free(Bloc_X);
        free(threads);

         print_info( "Jacobi ", CLASS, THREADS,time_end-time_start,
                  COMPILETIME, CC, CLINK, C_LIB, C_INC, CFLAGS, CLINKFLAGS );



}


double Distance(double *X_Old, double *X_New, int matrix_size)
{
        int             index;
        double          Sum;

        Sum = 0.0;
        for (index = 0; index < matrix_size; index++)
        Sum += (X_New[index] - X_Old[index]) * (X_New[index] - X_Old[index]);
        return (Sum);
}


void jacobi(int Number)
{
   int i,j;

   for(i = 0; i < Number; i++)
   {
     Bloc_X[i] = Input_B[i];

     for (j = 0;j<i;j++) 
     {
      Bloc_X[i] -= X_Old[j] * Matrix_A[i][j];
     }
     for (j = i+1;j<Number;j++) 
     {
     Bloc_X[i] -= X_Old[j] * Matrix_A[i][j];
     }
     Bloc_X[i] = Bloc_X[i] / Matrix_A[i][i];
  }
  for(i = 0; i < Number; i++)
  { 
  X_New[i] = Bloc_X[i];
  }
}  
