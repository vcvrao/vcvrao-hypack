
/********************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013
Example              : MatrixMatrixMultiplication.c
Objective            : Find out most time consuming section of this code 
                       using VTune Performance Analyzer, try to reduce execution 
                       time

                       To compute the matrix-matrix multiplication with 'p' 
                       threads using Self Scheduling algorithm.
                       Matirx is populated internally with 1 and 2 as elements.

                       Demonstrates use of:
                       pthread_create()
                       pthread_join()
                       pthread_mutex_t
                       pthread_mutex_lock()
                       pthread_mutex_unlock()
		           pthread_attr_init()
		           pthread_attr_destroy()

Input               : Sizes of matrices to be multiplied.

Output              : Product of Matrix Multiplication.


Created             : August-2013

E-mail              : hpcfte@cdac.in     

********************************************************************/ 


/******************************************************************
This is raw program , with out any optimization. 
********************************************************************/ 


#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>



 /* Size of Vector and Matrix. */ 
int row1, col1, row2, col2, currentRow = 0, **InMat1, **InMat2, **ResMat;

pthread_t * threads;
pthread_attr_t pta;

int  numberOfThreads;

 /* Mutex for the currentRow. */ 
pthread_mutex_t mutex_Row = PTHREAD_MUTEX_INITIALIZER;

pthread_cond_t count_threshold_cv = PTHREAD_COND_INITIALIZER;


void *doMyWork(int Id)
{
	
	int i, j, myRow, cnt;
	
		while (1) 
                {
		
			
			pthread_mutex_lock(&mutex_Row);
		
			if (currentRow >= row1) 
                        {
				pthread_mutex_unlock(&mutex_Row);
				if (Id == 0)
				return ;
			
				pthread_exit(0);
		        } 
			myRow = currentRow;
		
			currentRow++;
		
			pthread_mutex_unlock(&mutex_Row);
		
		
			for (j = 0; j < col2; j++)
			for (i = 0; i < col1; i++)
			ResMat[myRow][j] += InMat1[myRow][i] * InMat2[i][j];
		
			
	    } 
}


int main(int argc, char *argv[])
{
	
	int i, j;

	if ( argc < 6 )
         {
	   printf("\n Insufficient argumets. \n Usage:");
	   printf(" exe row1 col1 row2 col2 threads.\n");
	   return 1;
	  }
		
		row1 = abs(atoi(argv[1]));
		col1 = abs(atoi(argv[2]));
		row2 = abs(atoi(argv[3]));
		col2 = abs(atoi(argv[4]));
		numberOfThreads = abs(atoi(argv[5]));
	
	if (col1 != row2)
         {
    	     printf("\n Cannot multiply matrices of given sizes. Aborting.");
             return 1;
	 }	

        if (numberOfThreads > row2)
        {
           printf("\nNumber of threads should not be more than the number of rows of second matrix or number of columns of First matrix");
           return 1;
        }
		
	printf("\n Row1: %d. Col1: %d, Row2: %d,  Col2: %d. Number: %d.\n", row1, col1, row2, col2, numberOfThreads);
	
		
		InMat1 = (int **) malloc(sizeof(int *) * row1);
		for (i = 0; i < row1; i++)
		InMat1[i] = (int *) malloc(sizeof(int) * col1);
	
		
		InMat2 = (int **) malloc(sizeof(int *) * row2);
		for (i = 0; i < row2; i++)
		InMat2[i] = (int *) malloc(sizeof(int) * col2);
	
		
		ResMat = (int **) malloc(sizeof(int *) * row1);
		for (i = 0; i < row1; i++)
		ResMat[i] = (int *) malloc(sizeof(int) * col2);
	
		
		 /* Populate the Matrices. */ 
		
		for (i = 0; i < row1; i++)
		for (j = 0; j < col1; j++)
		InMat1[i][j] = 1;
	
		
		for (i = 0; i < row2; i++)
		for (j = 0; j < col2; j++)
		InMat2[i][j] = 2;
	
		
		for (i = 0; i < row1; i++)
		for (j = 0; j < col2; j++)
		ResMat[i][j] = 0;
		
		threads = (pthread_t *) malloc(sizeof(pthread_t) * numberOfThreads);

		pthread_attr_init(&pta);
	
		
		 /* Start Distributing the work. */ 
		
		currentRow = 0;
	
		for (i = 0; i < numberOfThreads; i++)
			pthread_create(&threads[i], &pta, (void *(*) (void *)) doMyWork, (void *) (i + 1));
		
		
		for (i = 0; i < numberOfThreads; i++)
	 	  pthread_join(threads[i], NULL);

		pthread_attr_destroy(&pta);
	
	free(InMat1);
       free(InMat2);
       free(ResMat);

	
                 return 0;
} 
