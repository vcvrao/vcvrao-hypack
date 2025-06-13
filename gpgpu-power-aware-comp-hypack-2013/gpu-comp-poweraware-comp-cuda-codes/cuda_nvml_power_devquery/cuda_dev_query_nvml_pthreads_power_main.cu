/*
	Objective : To launch deviceQuery to measure startup power of a GPU
	program   : main program that create two threads for power measuring and lunching devicequery.	 
	author 	  : HPC-FTEG 
*/


#include<cuda_dev_query_nvml_power_kernel_define.h>


int main(int argc, char **argv)
{
        pthread_t thread[3];
        pthread_attr_t attr;
        int tid1 = 0;
        int tid2 = 1;
        int i;

        /* create joinable threads */
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
        pthread_create(&thread[0], &attr, watch_count, (void *)tid1);
        pthread_create(&thread[1], &attr, deviceQueryFunc, (void *)tid2);

        /* wait for all threads to complete */
        for (i =0 ; i < 2; i++)
        {
                pthread_join(thread[i], NULL);
        }
        /* destroy all objects */
        pthread_attr_destroy(&attr);
        return 0;
}

