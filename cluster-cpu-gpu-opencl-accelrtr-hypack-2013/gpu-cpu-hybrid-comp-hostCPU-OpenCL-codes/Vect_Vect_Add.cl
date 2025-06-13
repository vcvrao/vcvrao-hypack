 __kernel void vectVectAddKernel(__global  float *d_a, __global float *d_b, __global float *d_output) 
{

       int tid = get_global_id(0);
       d_output[tid] =  d_a[tid] + d_b[tid];
}
