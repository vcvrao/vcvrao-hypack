/***********************************************************************************

  Objective  :  Inifinity Norm of Matrix using Row Wise & Column Wise splitting,

                Matrix - Vector Multiplication ( Checkerboard Algorithm ),
    
                Matrix - Matrix Multiplication ( Self-Scheduling Algorithm ).


  Input      :  Number of Threads,
         
                Size of Matrix1( Number of Rows and Number of Columns ),
            
                Size of Matrix2( Number of Rows and Number of Columns ).

  Output     :  Infinity Norm of Matrix1 (Using Row-Wise & Column-Wise splitting),

                Product of Matrix Multiplication,

                Resultant Vector of Matrix-Vector Multplication.

************************************************************************************/

#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include<stdlib.h>
#include<sys/time.h>

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

pthread_mutex_t  mutex_norm = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t  CurRow_norm = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t  *mutex_Res;
pthread_mutex_t  mutex_col;

pthread_mutex_t mutex_Row = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t count_threshold_cv = PTHREAD_COND_INITIALIZER;

double   NormRow = 0 ,NormCol = 0;
double   *Res;
int      dist_row,dist_col;
int      iteration;

int      rmajor,cmajor;
int      perfect_square;
int      row1, col1, row2, col2, currentRow = 0;
double   **InMat1, **InMat2, **ResMat, *vec, **fvec, *rvec;
int      numberOfThreads;

void  * doRowWise(int myId)
{
       int   CurRow = 0;        
       int   iCol,myRowSum;
       int   mynorm;

       for (CurRow = ((myId - 1) * dist_row); CurRow <= ((myId * dist_row) - 1); CurRow++)
       {
         
            myRowSum = 0;
            for(iCol = 0 ;iCol < col1; iCol++)
            myRowSum += InMat1[CurRow][iCol];

          if(mynorm < myRowSum )
             mynorm = myRowSum;
       }

       pthread_mutex_lock(&mutex_norm);
       {
             if (NormRow < mynorm)
              NormRow = mynorm;
       }
       pthread_mutex_unlock(&mutex_norm);

       pthread_exit(NULL);

}

void *doColWise(int myId)
{
    int iRow;
    int CurCol = 0;     
    for (CurCol = ((myId - 1) * dist_col); CurCol <= ((myId * dist_col) - 1); CurCol++)
     for(iRow = 0 ;iRow < row1; iRow++)
         Res[iRow] += InMat1[iRow][CurCol];
    pthread_exit(NULL);
}

void * doMatselfsched(int Id)
{
  int i, j, myRow, cnt;
  while (1) 
  {
   pthread_mutex_lock(&mutex_Row);
   if (currentRow >= row1)
   {
   pthread_mutex_unlock(&mutex_Row);
   if (Id == 0)
   exit(0); 
   pthread_exit(0);
  }
  myRow = currentRow;
  currentRow++;
  pthread_mutex_unlock(&mutex_Row);
  for (j = 0; j < col2; j++)
  for (i = 0; i < col1; i++)
  ResMat[myRow][j] += InMat1[myRow][i] * InMat2[i][j];
 }
  pthread_exit(NULL);
}

void * domatvectmul(int id)
{
 int i,j,rcs,ccs;

 if (id > (perfect_square - 1))
 rcs = (id % perfect_square) * rmajor;
 else
 rcs = id * rmajor;

 ccs = (id / perfect_square) * cmajor;
 for (i = rcs; i < rcs + rmajor; i++)
 for (j = ccs; j < ccs + cmajor; j++)
 fvec[i][j] = InMat1[i][j] * vec[j];
 pthread_exit(NULL);
}

main(int argc, char *argv[])
{
  int            i, j,p,q;
  int            first_value_row,diff,matrix_size;
  double         time_start, time_end,memoryused=0.0;
  struct         timeval tv;
  struct         timezone tz;
  int            counter, irow, icol, numberOfThreads,ithread;  
  FILE           *fp;

  pthread_t      *threads_row;
  pthread_t      *threads_col;
  pthread_attr_t ptr,ptc;

  pthread_t      * threads;
  pthread_t      * p_threads;
  pthread_attr_t pta;
  pthread_attr_t attr;


  printf("\n\t\t---------------------------------------------------------------------------");
  printf("\n\t\t Centre for Development of Advanced Computing (C-DAC):  December-2006");
  printf("\n\t\t C-DAC Multi Core Benchmark Suite 1.0");
  printf("\n\t\t Email : betatest@cdac.in");
  printf("\n\t\t---------------------------------------------------------------------------");
  printf("\n\t\t Objective : Dense Matrix Computations (Floating-Point Operations)\n ");
  printf("\n\t\t Performance of Four different Matrix Computation suites");
  printf("\n\t\t Computation of Infinity Norm of a Square Matrix - Rowwise/Columnwise partitioning;");
  printf("\n\t\t Matrix into Vector Multiplication; Matrix into Matrix Multiplication");
  printf("\n\t\t..........................................................................\n");

  numberOfThreads = THREADS;
   matrix_size = CLASS_SIZE;
   row1 = matrix_size;
   col1 = matrix_size;
   row2 = matrix_size;
   col2 = matrix_size;

   printf("\n\t\t Input Parameters :");
   printf("\n\t\t CLASS           :  %c",CLASS);
   printf("\n\t\t Matrix Size     :  %d",matrix_size);
   printf("\n\t\t Threads         :  %d",numberOfThreads);
   printf("\n");

  if ((numberOfThreads != 1) && (numberOfThreads != 2) && (numberOfThreads != 4) && (numberOfThreads != 8))
   {
       printf("\n Number of Threads must be 1 or 2 or 4 or 8. Aborting ...\n");
       exit(0);
   }

  if(numberOfThreads > row1)
   {
       printf("\nNumber of threads should be <= %d",row1);
       exit(0);
   }
   perfect_square = (sqrt(numberOfThreads));
  if (col1 != row2)
   {
    printf("\n Cannot multiply matrices of given sizes for Self Scheduling Algorithem. Aborting.");
    exit(0);
   }
  if (numberOfThreads > row2)
  {   printf("\nNumber of threads should not be more than the number of rows of Matrix 2 or number of columns of Matrix 1");
      exit(0);
  }

  /*....Memory Allocation....*/
  
  InMat1 = (double **) malloc(sizeof(double) * row1);
  for (i = 0; i < row1; i++)
  InMat1[i] = (double *) malloc(sizeof(double) * col1);
  memoryused += (row1 * col1 * sizeof(double));

  InMat2 = (double **) malloc(sizeof(double) * row2);
  for (i = 0; i < row2; i++)
  InMat2[i] = (double *) malloc(sizeof(double) * col2);
  memoryused += (row2 * col2 * sizeof(double));
  
  ResMat = (double **) malloc(sizeof(double) * row1);
  for (i = 0; i < row1; i++)
  ResMat[i] = (double *) malloc(sizeof(double) * col2);
  memoryused += (row1 * col2 * sizeof(double));

  Res = (double *) malloc(sizeof(double) * row1);
  memoryused += (row1 * sizeof(double));
 
  fvec = (double **) malloc(row1 * sizeof(double));
  for (i = 0; i < row1; i++)
  fvec[i] = (double *) malloc(col1 * sizeof(double));

  memoryused += (row1 * sizeof(double));

  vec = (double *) malloc(col1 * sizeof(double));
  memoryused += (col1 * sizeof(double));

  rvec = (double *) malloc(row1 * sizeof(double));
  memoryused += (row1 * sizeof(double));

  /* Matrix  and Vector  Initialization */
   for (i=0; i<row1;i++)
  {
   for (j = 0; j<col1;j++)
   {
   InMat1[i][j] = (double)(i + j);
   fvec[i][j] = 0.0;
   vec[j] = 2.0;
   Res[i] = 0.0;
   }
  }

  for (i = 0; i < row2; i++)
  for (j = 0; j < col2; j++)
  InMat2[i][j] = 1.0;
  for (i = 0; i < row1; i++)
  for (j = 0; j < col2; j++)
  ResMat[i][j] = 0.0;


  
 /* InfinityNorm Of InMat1 Matrix */

   /* Row Wise Partitioning */

  dist_row = row1/numberOfThreads;

  threads_row = (pthread_t *) malloc(sizeof(pthread_t) * numberOfThreads);

  gettimeofday(&tv, &tz);
  time_start = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;

  pthread_attr_init(&ptr);
  for (ithread = 0; ithread < numberOfThreads; ithread++)
  pthread_create(&threads_row[ithread], &ptr, (void *(*) (void *)) doRowWise, (void *) (ithread+1));

  for (ithread = 0; ithread < numberOfThreads; ithread++)
  pthread_join(threads_row[ithread], NULL);

  printf("\n\t\t Row Wise partitioning - Infinity Norm of a Square Matrix.....Done ");

  pthread_attr_destroy(&ptr);

  /* Column Wise Partitioning */

  dist_col = col1 / numberOfThreads;

  pthread_attr_init(&ptc);
  mutex_Res = ( pthread_mutex_t * ) malloc(sizeof(pthread_mutex_t) * row1);
  threads_col = ( pthread_t * ) malloc(sizeof(pthread_t) * numberOfThreads);

  for (ithread = 0; ithread < numberOfThreads; ithread++)
  pthread_create(&threads_col[ithread], &ptc, (void*(*)(void*)) doColWise, (void*) (ithread + 1));

  for (counter=0; counter<numberOfThreads; counter++)
  pthread_join(threads_col[counter], NULL);

  for (irow = 0 ; irow < row1 ; irow++)
  if (Res[irow] > NormCol)
    NormCol = Res[irow];

  printf("\n\t\t Row Wise partitioning - Infinity Norm of a Square Matrix.....Done ");
  pthread_attr_destroy(&ptc);

  /* Matrix-Matrix Multiplication using Self-Scheduling Method */
 
  threads = (pthread_t *) malloc(sizeof(pthread_t) * numberOfThreads);
  pthread_attr_init(&pta);

 /* Start Distributing the work. */

  currentRow = 0;
  for (i = 0; i < numberOfThreads; i++)
  pthread_create(&threads[i], &pta, (void *(*) (void *)) doMatselfsched, (void *) (i + 1));

  for (i = 0; i < numberOfThreads; i++)
  pthread_join(threads[i], NULL);
  pthread_attr_destroy(&pta);
  printf("\n\t\t Matrix-Matrix Multiplication using Self-Scheduling Method.....Done ");

 /* Matrix - Vector Multiplication using Checkerboard Algorithem */
   if((numberOfThreads != 4) && (numberOfThreads != 1))
   {
      goto exit;
   }
   p_threads = (pthread_t *) malloc(numberOfThreads * sizeof(pthread_t));
   rmajor = row1 / perfect_square;
   cmajor = col1 / perfect_square;
   pthread_attr_init(&attr);
   for (p = 0; p < numberOfThreads; p++)
   {
    pthread_create(&p_threads[p], &attr, (void *(*) (void *)) domatvectmul, (void *) p);
   }

   for (p = 0; p < numberOfThreads; p++)
   pthread_join(p_threads[p], NULL);
   pthread_attr_destroy(&attr);
   printf("\n\t\t Matrix - Vector Multiplication using Checkerboard Algorithm.....Done");

   free(InMat1);
   free(InMat2);
   free(ResMat);
   free(fvec);
   free(rvec);
   free(threads_row);
   free(threads_col);
   free(mutex_Res);
   free(Res);
   free(p_threads);
   free(threads);
   exit:
   gettimeofday(&tv, &tz);
   time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
   printf("\n");
   printf("\n\t\t Memory Utilized       :  %lf MB",(memoryused/(1024*1024)));
   printf("\n\t\t Time in  Seconds (T)  :  %lf",(time_end - time_start));
   printf("\n\t\t   ( T represents the Time taken to execute the four suites )\n");
   printf("\n\t\t..........................................................................\n");

print_info( "mat_cmp_db", CLASS, THREADS,time_end-time_start,
                  COMPILETIME, CC, CLINK, C_LIB, C_INC, CFLAGS, CLINKFLAGS );

}
