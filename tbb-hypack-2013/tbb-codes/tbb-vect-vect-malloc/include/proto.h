
#ifndef PROTO_H
#define PROTO_H 1

extern int vsize;                              // variable declaration

extern float *vectorB;
extern float *vectorA;
extern float *result_vector;

void vec_memory_allocation(float **,float **,float **);
void vector_input (float *,float *);
void par_vector_vector_multiply ();
void print_output(int,int,double);
void memoryfree(float *,float *,float *);

#endif
