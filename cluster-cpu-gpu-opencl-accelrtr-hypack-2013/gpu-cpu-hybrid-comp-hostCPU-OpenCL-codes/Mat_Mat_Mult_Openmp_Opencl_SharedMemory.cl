#define BLOCK_SIZE 16 
__kernel void matMatMultKernel( __global float* C, __global float* A, __global float* B, __local float* As, __local float* Bs, __global int* width_a, __global int* width_b)
{
    int blockIdx = get_group_id(0);
    int blockIdy = get_group_id(1);
    int threadIdx = get_local_id(0);
    int threadIdy = get_local_id(1);
    int aBegin = (*width_a) * BLOCK_SIZE * blockIdy;
    int aEnd   = aBegin + (*width_a) - 1;
    int aStep  = BLOCK_SIZE;
    int bBegin = BLOCK_SIZE * blockIdx;
    int bStep  = BLOCK_SIZE * (*width_b);
    float Csub = 0.0f;
    for (int a = aBegin, b = bBegin; a <= aEnd;a += aStep, b += bStep)
        {

        As[threadIdx + threadIdy * BLOCK_SIZE] = A[a + (*width_a) * threadIdy + threadIdx];
        Bs[threadIdx + threadIdy * BLOCK_SIZE] = B[b + (*width_b) * threadIdy + threadIdx];
        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k)
            Csub += As[k + threadIdy * BLOCK_SIZE] * Bs[threadIdx + k *BLOCK_SIZE];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    C[get_global_id(1) * get_global_size(0) + get_global_id(0)] = Csub;

}


