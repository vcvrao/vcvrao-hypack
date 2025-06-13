
#ifndef PROTO_H
#define PROTO_H 1
extern int nrows;
extern int ncols;
extern int vsize;                              // variable declaration

extern float *matrixA;
extern float *vectorA;
extern float *result_vector;


void  mat_vec_memory_allocation(float ** ,float **,float **);
void matrix_vector_input (float *,float *);
void par_matrix_vector_multiply ();
void print_output(int,int,double);
void memoryfree(float *,float *,float *);


#endif
