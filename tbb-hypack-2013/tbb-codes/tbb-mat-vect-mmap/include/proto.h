

#ifndef PROTO_H
#define PROTO_H 1

extern int nrows;
extern int ncols;
extern int vsize;                              // variable declaration

extern float *matrixA;
extern float *vectorA;
extern float *result_vector;


extern int fda, fdb, fdc;

float *  mmap_matvec_mem_allocation(char *,int,int, int, int*,int);
void mmap_matrix_input(char *);
void mmap_vector_input(char *);
void par_matrix_vector_multiply ();
void print_output(int,int,double);
void memoryfree(float *,float *,float *,size_t,size_t);


#endif



