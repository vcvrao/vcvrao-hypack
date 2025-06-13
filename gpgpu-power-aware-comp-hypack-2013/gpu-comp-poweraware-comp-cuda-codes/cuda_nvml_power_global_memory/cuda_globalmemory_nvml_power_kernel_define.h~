#ifndef POWER_H
#define POWER_H


/**
 * Include header files
**/
#include<stdio.h>
#include<pthread.h>
#include <stdlib.h>
#include <unistd.h>
#include<assert.h>
#include<cuda.h>


/** 
 * Holds epsilion value
**/
#define EPS 1.0e-15     

/**
 * cuda safe call
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
 * Function prototypes
**/
extern "C" void *CoalescedGlobalMemAccessFunc(void *t);
extern "C" void *watch_count(void *t);




#endif
