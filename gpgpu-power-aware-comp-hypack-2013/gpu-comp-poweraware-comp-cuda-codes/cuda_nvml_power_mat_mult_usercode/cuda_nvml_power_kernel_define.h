#ifndef POWER_H
#define POWER_H


/**
 * declare all header files
**/
#include<stdio.h>
#include<pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include<assert.h>
#include<cuda.h>


/**
 * define epsilion value
**/
#define EPS 1.0e-15  

/**
 * define block size of a matrix while
 * using shared memory.
 *
**/
#define BLOCKSIZE 16

/**
 * Define matrix size 
 *
**/
#define SIZE 10240


/**
 * cuda safe call 
 *
**/
#define CUDA_SAFE_CALL(call){\
                cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(-1);                                                  \
    }}\


/**
 * globalvariable used by thread
 * to send completion of work signal 
 * to another thread who is involved 
 * measuring power.
**/
extern int sigFlag;

/**
 * function prototypes
**/
extern "C" void *mat_mult(void *t);
extern "C" void *watch_count(void *t);


/* global variables for power_test kernel*/


#endif
