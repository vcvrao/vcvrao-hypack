

extern int vsize;

extern float *vectorB;                             // variable declaration
extern float *vectorA;
extern float *result_vector;
extern int mapvsize;

extern int fda, fdb, fdc;

float* mmap_vec_mem_allocation(char *, int, int, int *);
void mmap_vector_input (char *);
void par_vector_vector_multiply ();
void print_output(int,int,double);
void memoryfree(float *,float *,float *,size_t);
