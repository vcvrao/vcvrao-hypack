/*****************************************************************************************************
 			C-DAC Tech Workshop : hyPACK-2013
                        	 October 15-18, 2013

 Example           :  mat-mat-mul-openmp-coprocessor.c
			     
 Objective         :  Matrix Matrix Multiplication using openMP on Xeon Phi Coprocessor (offload, native)

 Input             :  Set <no. of mic threads> <square matrix size> <alignment>

 Output            :  Print the Time Elapsed and GFLOPS/s
 		
 		      Offload:
 		      1) Serial matrix multiplication with openmp on host
 		      2) Transpose matrix multiplication with openmp on host
 		      3) Offload matrix multiplication with openmp on coprocessor

		      Native:
		      1) All three funtions run on coprocessor (offload disabled)

 Created           :  August-2013

 E-mail            :  hpcfte@cdac.in   

*****************************************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <offload.h>
#include <sys/time.h>
#include <time.h>

#ifndef MIC_DEV
#define MIC_DEV 0
#endif

struct matMulSetup
{
        int hostCores;
	int matrixSz;
        int alignment;
};

void matMatMul_MIC_Offload(struct matMulSetup * setup, double * matA, double * matB, double * matC);
void matMatMul_MIC_Host(struct matMulSetup * setup, double * matA, double * matB, double * matC);
void matMatMul_MIC_Serial(struct matMulSetup * setup, double * matA, double * matB, double * matC);


void simpleMatrixMultiplication(double * MatrixA,  int rowASize, int colASize, double * MatrixB, int colBSize, double * resultMatrix);
void writeToFile(double * resultMatrix, unsigned int row, unsigned int col, char * fileName);
int allocateMatrixMemory1D(double **  matrix, unsigned int row, unsigned int col, unsigned int alignment);
int generateMatrixData(double * matrix, unsigned int row, unsigned int col, unsigned int range);
void freeHeapSpace1D(double ** matrix);
int transposeMatrix(double * matrix, double * transposeMatrix,  unsigned int row, unsigned int col);
void displayMatrix(double * matrix, int row, int col);
int zeroInitialise(struct matMulSetup * setup, double * matrix);

int main(int argc, char ** argv)
{
	if(argc!=4)
        {
                printf("Invalid Commandline Argument Given.\n");
                printf("Usage: ./executableName.out #micThreads matrixSz alignment\n");
                return 1;
        }

	struct matMulSetup setup;
	
        setup.hostCores=atoi(argv[1]);
	setup.matrixSz=atoi(argv[2]);
        setup.alignment=atoi(argv[3]);

	double * matA=NULL, * matB=NULL, * matC=NULL;

        if(allocateMatrixMemory1D(&matA, setup.matrixSz, setup.matrixSz, setup.alignment))
        {
                exit(-1);
        }     

        if(allocateMatrixMemory1D(&matB, setup.matrixSz, setup.matrixSz, setup.alignment))
        {
                exit(-1);
        }     
        
	if(allocateMatrixMemory1D(&matC, setup.matrixSz, setup.matrixSz, setup.alignment))
        {
                exit(-1);
        }   

	generateMatrixData(matA, setup.matrixSz, setup.matrixSz, 100);
	generateMatrixData(matB, setup.matrixSz, setup.matrixSz, 100);

	zeroInitialise(&setup, matC);
	
	matMatMul_MIC_Serial(&setup, matA, matB, matC);
	writeToFile(matC, setup.matrixSz, setup.matrixSz, "serialMul.r"); 	

	zeroInitialise(&setup, matC);

	matMatMul_MIC_Host(&setup, matA, matB, matC);
	writeToFile(matC, setup.matrixSz, setup.matrixSz, "transposeMul.r"); 	
	 
	zeroInitialise(&setup, matC);

	matMatMul_MIC_Offload(&setup, matA, matB, matC);
	writeToFile(matC, setup.matrixSz, setup.matrixSz, "offloadMul.r"); 	

	freeHeapSpace1D(&matA);
        freeHeapSpace1D(&matB);
        freeHeapSpace1D(&matC);
        matA=matB=matC=NULL;
 
	return 0;
}

void matMatMul_MIC_Offload(struct matMulSetup * setup, double * __restrict__  matA, double * __restrict__ matB, double * __restrict__ matC)
{
	int size=setup->matrixSz;
	float (*restrict Matrix_A)[size] = malloc(sizeof(float)*size*size);
 	float (*restrict Matrix_B)[size] = malloc(sizeof(float)*size*size);
  	float (*restrict Matrix_C)[size] = malloc(sizeof(float)*size*size);
    
	#pragma omp parallel for default(none) shared(Matrix_A, Matrix_B,size, matA, matB) 
   	for(int i = 0; i < size; ++i) 
	{
       		for(int j = 0; j < size; ++j) 
		{
        	 	Matrix_A[i][j] = matA[i*size+j]; //(float)i + j;
         		Matrix_B[i][j] = matB[i*size+j]; //(float)i - j;
    		}
	}

 	struct timeval tim;
        printf("Performing Mic Offloaded Matrix Multiplication\n");
        gettimeofday(&tim, NULL);
        double startTime=tim.tv_sec+(tim.tv_usec*1E-6), gflops=0.0F, diffTime=0.0F;
        double opCount=2.0F*setup->matrixSz*setup->matrixSz*setup->matrixSz*1E-9;

	#pragma offload target(mic:MIC_DEV) \
      	in(Matrix_A:length(size*size)) \
      	in(Matrix_B:length(size*size)) \
     	out(Matrix_C:length(size*size))
   	{

     		 #pragma omp parallel for default(none) shared(Matrix_C,size) 
		 for(int i = 0; i < size; ++i)
      			for(int j = 0; j < size; ++j)
         			Matrix_C[i][j] =0.0F;

  		#pragma omp parallel for default(none) shared(Matrix_A,Matrix_B,Matrix_C,size) 
     		for (int i = 0; i < size; ++i)
     			for (int k = 0; k < size; ++k)
			{
     				for (int j = 0; j < size; ++j)
				{
      					Matrix_C[i][j] += Matrix_A[i][k] * Matrix_B[k][j];
				}
			}
		
	}

        gettimeofday(&tim, NULL);
        printf("Mic Offloaded Matrix Multiplication Done\n");
        double endTime=tim.tv_sec+(tim.tv_usec*1E-6);
        diffTime=endTime-startTime;
        printf("%.6lf seconds elapsed\n", diffTime);
        gflops=opCount*(1.0F/diffTime);
        printf("%.6lf gflops\n\n", gflops);
      
     	#pragma omp parallel for default(none) shared(matC, Matrix_C,size) 
   	for(int i = 0; i < size; ++i) 
	{
       		for(int j = 0; j < size; ++j) 
		{
         		//matC[i*size+j]=Matrix_C[i][j];
         		double x=matC[i*size+j]-Matrix_C[i][j];
         		if(x>0.01)
			{
				//printf("%lf, %lf\n", matC[i*size+j], Matrix_C[i][j]);
			}
         		
    		}
	}

 	free(Matrix_A);
	free(Matrix_B);
  	free(Matrix_C);
}


void matMatMul_MIC_Host(struct matMulSetup * setup, double * matA, double * matB, double * matC)
{
	double * matBT=NULL;

        if(allocateMatrixMemory1D(&matBT, setup->matrixSz, setup->matrixSz, setup->alignment))
        {
                exit(-1);
        }    

		
 	struct timeval tim;
        printf("Performing Transpose Matrix Multiplication\n");
        gettimeofday(&tim, NULL);
        double startTime=tim.tv_sec+(tim.tv_usec*1E-6), gflops=0.0F, diffTime=0.0F;
        double opCount=2.0F*setup->matrixSz*setup->matrixSz*setup->matrixSz*1E-9;
	
	transposeMatrix(matB, matBT, setup->matrixSz, setup->matrixSz);

        int i=0, j=0, k=0, l=0;
	int rowASize=setup->matrixSz, colASize=setup->matrixSz, colBSize=setup->matrixSz;
        double result=0.0;

	#pragma omp parallel for shared(matA, matB, matC) private(j, k) firstprivate(result) num_threads(12)
        for(i=0; i<rowASize; i++)
        {
                for(j=0; j<colBSize; j++)
                {
                        result=0.0;
                        for(k=0; k<colASize; k++)
                        {
                                result+=matA[colASize*i+k]*matBT[colBSize*j+k];
                        }
                        matC[colASize*i+j]=result;
                }
        }

        gettimeofday(&tim, NULL);
        printf("Transpose Matrix Multiplication Done\n");
        double endTime=tim.tv_sec+(tim.tv_usec*1E-6);
        diffTime=endTime-startTime;
        printf("%.6lf seconds elapsed\n", diffTime);
        gflops=opCount*(1.0F/diffTime);
        printf("%.6lf gflops\n\n", gflops);
        freeHeapSpace1D(&matBT);
	
	//printf("%lf\n", matC[0]);
}

void matMatMul_MIC_Serial(struct matMulSetup * setup, double * matA, double * matB, double * matC)
{
 	struct timeval tim;
        printf("Performing Serial Matrix Multiplication\n");
        gettimeofday(&tim, NULL);
        double startTime=tim.tv_sec+(tim.tv_usec*1E-6), gflops=0.0F, diffTime=0.0F;
        double opCount=2.0F*setup->matrixSz*setup->matrixSz*setup->matrixSz*1E-9;

        int i=0, j=0, k=0;
	int rowASize=setup->matrixSz, colASize=setup->matrixSz, colBSize=setup->matrixSz;
        double result=0.0;

	#pragma omp parallel for shared(matA, matB, matC) private(j, k) firstprivate(result) num_threads(12)
        for(i=0; i<rowASize; i++)
        {
                for(j=0; j<colBSize; ++j) 
                {
                        result=0.0;
                        for(k=0; k<colASize; k++)
                        {
                                result+=matA[colASize*i+k]*matB[colBSize*k+j];
                        }
                        matC[colASize*i+j]=result;
                }
        }

        gettimeofday(&tim, NULL);
        printf("Serial Matrix Multiplication Done\n");
        double endTime=tim.tv_sec+(tim.tv_usec*1E-6);
        diffTime=endTime-startTime;
        printf("%.6lf seconds elapsed\n", diffTime);
        gflops=opCount*(1.0F/diffTime);
        printf("%.6lf gflops\n\n", gflops);

}

void writeToFile(double * resultMatrix, unsigned int row, unsigned int col, char * fileName)
{
	if(!fileName)
        {

                printf("Invalid File Name.\n");
                return;
        }

        FILE * resultFile=fopen(fileName, "w");
        if(!resultFile)
        {
                printf("Output File for storing Matrix Multiplication result, could not be open.\n");
                return;
        }


        for(int i=0; i<row*col; i++)
        {
                if(i!=0 && i%col==0)
                {
                        fprintf(resultFile, "\n%f\t", resultMatrix[i]);
                }
                else
                {
                        fprintf(resultFile, "%f\t", resultMatrix[i]);
                }
        }
        fclose(resultFile);
}

int allocateMatrixMemory1D(double ** matrix, unsigned int row, unsigned int col, unsigned int alignment)
{
        *matrix=(double *)_mm_malloc(sizeof(double)*row*col, alignment);

        if(!(*matrix))
        {
                printf("Memory Allocation Failure for Matrix.");
                return 1;
        }
        return 0;
}

int generateMatrixData(double * matrix, unsigned int row, unsigned int col, unsigned int range)
{
        if(!matrix)
                return 1;

        srand((unsigned)time(NULL));
        int i=0;
        double randomNumber=0.0;
        for(i=0; i<row*col; i++)
        {
                randomNumber = rand() / (RAND_MAX + 1.0);
                matrix[i]=randomNumber;// *range;
                //matrix[i]=1.0F+i;
		
	}
	return 0;
}	

int transposeMatrix(double * matrix, double * transposeMatrix, unsigned int row, unsigned int col)
{
        if(!matrix || !transposeMatrix)
                return 1;

        int i=0, j=0;

        for(i=0; i<row; i++)
        {
                for(j=0; j<col; j++)
                {
                        transposeMatrix[j*col+i]=matrix[i*col+j];
                }
        }
        return 0;
}

void freeHeapSpace1D(double ** matrix)
{
        if(matrix && *matrix)
        {
		_mm_free(*matrix);
        }
        *matrix=NULL;
}

void displayMatrix(double * matrix, int row, int col)
{
        int i=0, j=0;

        for(i=0; i<row*col; i++)
        {
                if(i!=0 && i%col==0)
                {
                        printf("\n%.1f\t", matrix[i]);
                }
                else
                {
                        printf("%.1f\t", matrix[i]);
                }
        }
        printf("\n");
}

int zeroInitialise(struct matMulSetup * setup, double * matrix)
{
	if(!matrix)
		return 1;

	int size=setup->matrixSz*setup->matrixSz;
	for(int i=0; i<size; ++i)
	{
		matrix[i]=0.0F;
	}

	return 0;		
}

