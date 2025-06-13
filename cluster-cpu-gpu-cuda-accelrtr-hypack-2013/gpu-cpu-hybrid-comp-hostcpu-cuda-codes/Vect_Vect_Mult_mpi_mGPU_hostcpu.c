/****************************************************************************
                C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

  Objective   : program to solve a Vector Vector multiplication using block
                striped partioning on hybrid computing using MPI and CUDA on Multi-GPU

  Input       : Process 0 initialize the Vector.

  Output      : Process 0 prints the resultant value.

  Necessary      Size of the  Each Vector should be greater than number of
  Conditons   :  processes and perfectly divisible by number of processes.

  Created     : August-2013

  E-mail      : hpcfte@cdac.in

****************************************************************************/

#include <mpi.h>

float gpu_partial_sum_calculation(int );
void hfree();
float compare();
int main(int argc, char *argv[])
{
        int rank, size;
	float partial_sum,res2,Root=0;
	int vlen;
        MPI_Init (&argc, &argv);        /* starts MPI */
        MPI_Comm_rank (MPI_COMM_WORLD, &rank);  /* get current process id */
        MPI_Comm_size (MPI_COMM_WORLD, &size);  /* get number of processes */
	
	if(rank==0)
	{
	
		MPI_Status status;
                //float a=call(rank);
                float result=gpu_partial_sum_calculation(rank+1);
                partial_sum=result;
                printf("%f\n",partial_sum);
	
		int iproc,Source,Source_tag;
		float value,sum=partial_sum;
		for(iproc = 1 ; iproc <=1 ; iproc++)
		{
           		
			Source     = iproc;
	   		Source_tag = 0;
			
	   		MPI_Recv(&value, 1, MPI_FLOAT, Source, Source_tag,  MPI_COMM_WORLD, &status);
			sum=sum+value;
		//	printf("sum=%f",sum);
		}
		printf("final result of multiplication of vectors is:%f\n",sum);
		float cpu_result=compare();
		if(cpu_result!=sum)
			printf("cpu and gpu results dont match :( :( \n");
		else
			printf("***********cpu and gpu results matched :) :) \n");
		hfree();
	}
	 if(rank==1  )
        {
                MPI_Status status;
                //float a=call(rank);
                float result=gpu_partial_sum_calculation(rank+1);
                partial_sum=result;
                printf("%f\n",partial_sum);
                int Destination     = 0;
                int Destination_tag = 0;

                MPI_Send(&partial_sum, 1, MPI_FLOAT, Destination, Destination_tag,  MPI_COMM_WORLD);
        }

	 MPI_Finalize();
        return 0;
}

