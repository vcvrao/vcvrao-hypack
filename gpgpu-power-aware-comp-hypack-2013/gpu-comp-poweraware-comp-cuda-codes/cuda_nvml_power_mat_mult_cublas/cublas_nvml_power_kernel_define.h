#ifndef POWER_H
#define POWER_H

/**
 * define header files
**/
#include<stdio.h>
#include<pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include<assert.h>
#include<cuda.h>

/**
 * epsilion value for check result 
**/
#define EPS 1.0e-15    

/**
 * cuda safe call for
 * checking cuda APIs error.
**/
#define CUDA_SAFE_CALL(call){\
                cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        \
                __FILE__, __LINE__, cudaGetErrorString( err) );              \
        exit(-1);                                                  \
    }}\


/**
 * conditional variable 
**/
extern int sigFlag;

/**
 * function prototypes for measurinf power
 * and kernel part.
**/
extern "C" void *mat_mult(void *t);
extern "C" void *watch_count(void *t);


#endif
