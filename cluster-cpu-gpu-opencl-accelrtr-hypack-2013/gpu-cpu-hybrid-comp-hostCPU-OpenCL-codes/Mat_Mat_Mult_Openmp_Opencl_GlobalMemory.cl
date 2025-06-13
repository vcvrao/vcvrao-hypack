__kernel void matMatMultKernel(__global  float *d_a, __global float *d_b, __global float *d_output,__global int *width_a, __global int *width_b)
{
       int globalIdx = get_global_id(0);
       int globalIdy = get_global_id(1);
       int sum =0.0f,i,tempA,tempB;
       for ( i=0; i< (*width_a); i++)
       {
         tempA = d_a[globalIdy * (*width_a) + i];
         tempB = d_b[i * (*width_b) + globalIdx];
         sum += tempA * tempB;
        }
       d_output[globalIdy * (*width_b) + globalIdx] = sum;
}
