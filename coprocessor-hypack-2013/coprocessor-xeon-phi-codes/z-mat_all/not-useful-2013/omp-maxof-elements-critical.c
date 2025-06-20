/***********************************************************************************
 			C-DAC Tech Workshop : HeGaPa-2012
                             July 16-20,2012




  Example 3.1 : omp-maxof-elements-critical.c
 
  Objective   : Write an OpenMP program to print Largest of an element in
                an array
                This example demonstrates the use of
                omp_critical section call and PARALLEL For DIRECTIVE


  Input       : Number of threads 
                Number of elements of the array Input
                is generated by random numbers


  Output      : Each thread checks with its available iterations and finally
                Master thread prints the maximum value in the array ,Time taken
                to find the Max Element and also the threads .

	                                                                                                                    
  Created     : MAY-2012        

  E-mail      : betatest@cdac.in                
                          
****************************************************************************/

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <sys/time.h>
#define MAXIMUM 65536

/* Main Program */
main(int argc,char **argv)
{
	int            *array, i, Noofelements, cur_max, current_value,Noofthreads;
	struct timeval  TimeValue_Start;
        struct timezone TimeZone_Start;

        struct timeval  TimeValue_Final;
        struct timezone TimeZone_Final;
        long            time_start, time_end;
        double          time_overhead;


 	printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Centre for Development of Advanced Computing (C-DAC)");
        printf("\n\t\t Email : betatest@cdac.in");
        printf("\n\t\t---------------------------------------------------------------------------");
        printf("\n\t\t Objective : Finding Maximum element of an Array using  ");
        printf("\n\t\t OpenMP Parallel for directive and Critical Section  ");
        printf("\n\t\t..........................................................................\n");

         /* Checking for command line arguments */
        if( argc != 3 ){

           printf("\t\t Very Few Arguments\n ");
           printf("\t\t Syntax : exec <Threads> <No. of elements> \n");
           exit(-1);
        }

        Noofthreads=atoi(argv[1]);
        if ((Noofthreads!=1) && (Noofthreads!=2) && (Noofthreads!=4) && (Noofthreads!=8) && (Noofthreads!= 16) ) {
               printf("\n Number of threads should be 1,2,4,8 or 16 for the execution of program. \n\n");
               exit(-1);
         }


         Noofelements=atoi(argv[2]); 

/*	printf("\n\t\t Enter the number of elements\n");
	scanf("%d", &Noofelements);*/

	if (Noofelements <= 0) {
		printf("\n\t\t The array elements cannot be stored\n");
		exit(1);
	}
        printf("\n\t\t Threads                     : %d ",Noofthreads);
        printf("\n\t\t Number of elements in Array : %d \n ",Noofelements);
	/* Dynamic Memory Allocation */
	array = (int *) malloc(sizeof(int) * Noofelements);

	/* Allocating Random Number Values To The Elements Of An Array */
	srand(MAXIMUM);
	for (i = 0; i < Noofelements; i++)
		array[i] = rand();

	if (Noofelements == 1) {
		printf("\n\t\t The Largest Number In The Array is %d", array[0]);
		exit(1);
	}
	
	cur_max = 0;
        gettimeofday(&TimeValue_Start, &TimeZone_Start);
 
        /* Set the No. of threads */
	omp_set_num_threads(Noofthreads);
	
	/* OpenMP Parallel For Directive And Critical Section : Fork a team of threads */
	#pragma omp parallel for
	for (i = 0; i < Noofelements; i = i + 1) {
		if (array[i] > cur_max)
	#pragma omp critical
			if (array[i] > cur_max)
				cur_max = array[i];
	} /* End of the parallel section */

	 gettimeofday(&TimeValue_Final, &TimeZone_Final);

        /* calculate the timing for the computation */
        time_start = TimeValue_Start.tv_sec * 1000000 + TimeValue_Start.tv_usec;
        time_end = TimeValue_Final.tv_sec * 1000000 + TimeValue_Final.tv_usec;
        time_overhead = (time_end - time_start)/1000000.0;

	/* Serial Calculation */

	current_value = array[0];
	for (i = 1; i < Noofelements; i++)
		if (array[i] > current_value)
			current_value = array[i];

	/* Checking For Output Validity */

	if (current_value == cur_max)
		printf("\n\t\t The Max Value Is Same From Serial And Parallel OpenMP Directive\n");
	else {
		printf("\n\t\t The Max Value Is Not Same In Serial And Parallel OpenMP Directive\n");
		exit(-1);
	}

	/* Freeing Allocated Memory */

	printf("\n");
	free(array);
	printf("\n\t\t The Largest Number In The Given Array Is %d\n", cur_max);
        printf("\n\t\t Time in Seconds (T)        : %lf Seconds \n",time_overhead);
        printf("\n\t\t..........................................................................\n");
}
