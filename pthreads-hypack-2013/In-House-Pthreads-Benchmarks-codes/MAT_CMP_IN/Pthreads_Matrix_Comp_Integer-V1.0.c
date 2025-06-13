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
#define  CLASS_SIZE 1024 
#endif

#if CLASS == 'B'
#define  CLASS_SIZE  2048
#endif

#if CLASS == 'C'
#define  CLASS_SIZE 4096
#endif


pthread_mutex_t  mutex_norm = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t  CurRow_norm = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_Row = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t count_threshold_cv = PTHREAD_COND_INITIALIZER;

pthread_mutex_t  *mutex_Res;
pthread_mutex_t  mutex_col;


int      NormRow = 0 ,NormCol = 0;
int      *Res;
int      dist_row,dist_col;
int      iteration;
int      rmajor,cmajor;
int      perfect_square;
int      row1, col1, row2, col2, currentRow = 0;
int      **InMat1, **InMat2, **ResMat, *vec, **fvec, *rvec,**ResultMat,**finalvec;
int      numberOfThreads;
int      infinitynorm_row,infinitynorm_col; 

void  result_cal(int global_matrix_size)
{
   int i,j,row_sum,col_sum,k;
   infinitynorm_row=0;infinitynorm_col=0;

   /* Infinity norm row_wise calculation by Master thread only */
   for(i=0;i<global_matrix_size;i++)
   {
        row_sum=0;
        for(j=0;j<global_matrix_size;j++)
        row_sum += InMat1[i][j]; 
        if(infinitynorm_row < row_sum )
        infinitynorm_row=row_sum;
    }

   /* Infinity norm col_wise calculation by Master thread only */
    for(j=0;j<global_matrix_size;j++)
    {
      col_sum=0;
      for(i=0;i<global_matrix_size;i++)
      { 
         col_sum += InMat1[i][j];
      }
      if(infinitynorm_col < col_sum )
      infinitynorm_col = col_sum;
    }      

   /* Matrix Matrix multiplication by Master thread only */
    for(i=0;i<global_matrix_size;i++)
    for(j=0;j<global_matrix_size;j++)
    ResultMat[i][j] =0; 
    for(k=0;k<global_matrix_size;k++)
    {
     for(i=0;i<global_matrix_size;i++)
     {
       for(j=0;j<global_matrix_size;j++)
       ResultMat[k][i] += InMat1[k][j] * InMat2[j][i]; 
     }
    }

   /* Matrix Vector multiplication by Master thread only */
    for(i=0;i<global_matrix_size;i++)
    {
     for(j=0;j<global_matrix_size;j++)
     {
      finalvec[i][j]=InMat1[i][j] * vec[j];
     }
    }  
}

void results_compare(int matrix_sizes)
{
    int i,j,count=0;
   printf("\n");
   /* Comparing the results for Row_Wise Infinity norm calculation */
   if(infinitynorm_row == NormRow)
   printf("\n\t\t Row Wise partitioning - Infinity Norm of a Square Matrix.....Successful");  
   else
   printf("\n\t\t Row Wise partitioning - Infinity Norm of a Square Matrix.....Unsuccessful");  
     
   
   /* Comparing the results for Col_Wise Infinity norm calculation */
   if(infinitynorm_col == NormCol)
   printf("\n\t\t Column Wise partitioning - Infinity Norm of a Square Matrix.....Successful");  
   else
   printf("\n\t\t Column Wise partitioning - Infinity Norm of a Square Matrix.....Unsuccessful");  
  
   
   /* Comparing the results for Matrix-Matrix Multiplication by self-sheduling Algorithm */
   for(i=0;i<matrix_sizes;i++)
   {
     for(j=0;j<matrix_sizes;j++)
     {
       if(ResultMat[i][j] != ResMat[i][j])
       count ++;
     }     
   }        
   if(count ==0)
   printf("\n\t\t Matrix-Matrix Multiplication using Self-Scheduling Method.....Successful");
   else
   printf("\n\t\t Matrix-Matrix Multiplication using Self-Scheduling Method.....Unsuccessful");
   count =0;
   
   /* Comparing the results for Matrix-Vector Multiplication by Checkerboard Algorithm */
   
   for(i=0;i<matrix_sizes;i++)
   {
     for(j=0;j<matrix_sizes;j++)
     {
       if(fvec[i][j] != finalvec[i][j])
       count ++;
     }     
   }        
   if(count ==0)
   printf("\n\t\t Matrix - Vector Multiplication using Checkerboard Algorithm.....Successful");
   else
   printf("\n\t\t Matrix - Vector Multiplication using Checkerboard Algorithm.....Unsuccessful");

}
   
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
  printf("\n\t\t Objective : Dense Matrix Computations (Integer Operations)\n ");
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
 
  if(numberOfThreads > row1)
   {
       printf("\n Error : Number of threads should be <= %d",row1);
       exit(0);
   }
   perfect_square = (sqrt(numberOfThreads));
  if (col1 != row2)
   {
    printf("\n Error : Cannot multiply matrices of given sizes for Self Scheduling Algorithem. Aborting.");
    exit(0);
   }
  if (numberOfThreads > row2)
  {   printf("\nError : Number of threads should not be more than the number of rows of Matrix 2 or number of columns of Matrix 1");
      exit(0);
  }

  /*....Memory Allocation....*/
  
  InMat1 = (int **) malloc(sizeof(int *) * row1);
  for (i = 0; i < row1; i++)
  InMat1[i] = (int *) malloc(sizeof(int) * col1);
  if(InMat1 ==NULL)
  {
  printf("\n Not sufficient Memory to accomodate InMat1\n");
  exit(0);
  }
  memoryused += (row1 * col1 * sizeof(int));

  InMat2 = (int **) malloc(sizeof(int *) * row2);
  for (i = 0; i < row2; i++)
  InMat2[i] = (int *) malloc(sizeof(int) * col2);
  if(InMat2 ==NULL)
  {
  printf("\n Not sufficient Memory to accomodate InMat2\n");
  exit(0);
  }
  memoryused += (row2 * col2 * sizeof(int));
  
  ResMat = (int **) malloc(sizeof(int *) * row1);
  for (i = 0; i < row1; i++)
  ResMat[i] = (int *) malloc(sizeof(int) * col2);
  if(ResMat ==NULL)
  {
  printf("\n Not sufficient Memory to accomodate ResMat\n");
  exit(0);
  }
  memoryused += (row1 * col2 * sizeof(int));

  ResultMat = (int **) malloc(sizeof(int *) * row1);
  for (i = 0; i < row1; i++)
  ResultMat[i] = (int *) malloc(sizeof(int) * col2);
  if(ResultMat ==NULL)
  {
  printf("\n Not sufficient Memory to accomodate ResultMat\n");
  exit(0);
  }
  memoryused += (row1 * col2 * sizeof(int));

  Res = (int *) malloc(sizeof(int) * row1);
  if(Res ==NULL)
  {
  printf("\n Not sufficient Memory to accomodate Res\n");
  exit(0);
  }
  memoryused += (row1 * sizeof(int));
 
  fvec = (int **) malloc(row1 * sizeof(int *));
  for (i = 0; i < row1; i++)
  fvec[i] = (int *) malloc(col1 * sizeof(int));
  if(fvec ==NULL)
  {
  printf("\n Not sufficient Memory to accomodate fvec\n");
  exit(0);
  }

  memoryused += (row1 * sizeof(int));

  finalvec = (int **) malloc(row1 * sizeof(int *));
  for (i = 0; i < row1; i++)
  finalvec[i] = (int *) malloc(col1 * sizeof(int));
  if(finalvec ==NULL)
  {
  printf("\n Not sufficient Memory to accomodate finalvec\n");
  exit(0);
  }

  memoryused += (row1 * sizeof(int));

  vec = (int *) malloc(col1 * sizeof(int));
  if(vec ==NULL)
  {
  printf("\n Not sufficient Memory to accomodate vec\n");
  exit(0);
  }
  memoryused += (col1 * sizeof(int));

  rvec = (int *) malloc(row1 * sizeof(int));
  if(rvec ==NULL)
  {
  printf("\n Not sufficient Memory to accomodate rvec\n");
  exit(0);
  }
  memoryused += (row1 * sizeof(int));

  /* Matrix  and Vector  Initialization */
   for (i=0; i<row1;i++)
  {
   for (j = 0; j<col1;j++)
   {
   InMat1[i][j] = i+j;
   fvec[i][j] = 0;
   vec[j] = 2;
   Res[i] = 0;
   }
  }

  for (i = 0; i < row2; i++)
  for (j = 0; j < col2; j++)
  InMat2[i][j] = 1;
  for (i = 0; i < row1; i++)
  for (j = 0; j < col2; j++)
  ResMat[i][j] = 0;
 
  result_cal(matrix_size);
  
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

  printf("\n\t\t Column Wise partitioning - Infinity Norm of a Square Matrix.....Done ");
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
   free(ResultMat);
   free(finalvec);
   exit:
   results_compare(matrix_size);
   gettimeofday(&tv, &tz);
   time_end = (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
   printf("\n");
   printf("\n\t\t Memory Utilized        :  %lf MB",(memoryused/(1024*1024)));
   printf("\n\t\t Time in  Seconds (T)   :  %lf",(time_end - time_start));
   printf("\n\t\t   ( T represents the Time taken to execute the four suites )"); 
  printf("\n\t\t..........................................................................\n");
  print_info( "mat_cmp_in", CLASS, THREADS,time_end-time_start,
                  COMPILETIME, CC, CLINK, C_LIB, C_INC, CFLAGS, CLINKFLAGS );

}
