/*******************************************************************

		C-DAC Tech Workshop : hyPACK-2013
                           October 15-18, 2013

 Example              : Find-Prime.c

 Objective            : this program finds Number of primes number with in the range 
                        of 1 to 20000. But the result is contradictory with each other 
                        because of existance of data race condition at some specific 
			part of this program. Main objective of this program is to give 
                        an exposure to programmers to data race condition and how to detect 
                        it through intel thread checker.
 
 Input                : none

 Output               : Number of prime number with in range of 1 to 20000 	                                            
                                                                        

 Created              : August-2013

 E-mail               : hpcfte@cdac.in     


*********************************************************************************/

#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 4
#define MAX_NUMBERS 20000
#define BLOCKSIZE (MAX_NUMBERS / NUM_THREADS)

long primes[MAX_NUMBERS];
int primeCount;

void *findPrimes ( void * threadNum )
{
    long threadId   = *(long *)threadNum;
    long start  = threadId * BLOCKSIZE + 1;
    long end    = threadId * BLOCKSIZE + BLOCKSIZE;
    long stride = 2;
    long count, factor;

    if(start == 1) start += stride;
    for (count = start; count < end; count += stride )
    {
        factor = 3;
        while ( (count % factor) != 0 ) factor += 2;
        if ( factor == count )
        {
            primes[ primeCount ] = count;
            primeCount++;
        }
    }
   // return 0;
}

int main()
{
    int count, rc;
    long threadId[NUM_THREADS];
    pthread_t thread[NUM_THREADS];

    primeCount = 0;
    primes[primeCount++] = 2; 

    printf( "\n Number Of Prime Number with in 1 to %d range : ", MAX_NUMBERS);
    for ( count = 0; count < NUM_THREADS; ++count)
    {
        threadId[count] = count;
        rc = pthread_create (&thread[count], 0, findPrimes, (void *) &threadId[count]);
    }

    for ( count = 0; count < NUM_THREADS; ++count)
    {
        rc = pthread_join (thread[count], 0);
    }

    printf( "%d\n\n", primeCount );

    return 0;
}

