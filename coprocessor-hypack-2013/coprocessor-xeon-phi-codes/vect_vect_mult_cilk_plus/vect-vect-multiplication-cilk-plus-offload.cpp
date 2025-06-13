/**************************************************************
 *  
 *
 * FILE         : vect-vect-multiplication-cilk-plus.cpp 
 *    
 * Input        : Size of vector
 * 
 * Output       : Time elapsed to compute vector multiplication 
 *                Uses array notation of cilkplus       
 *                                    
 *              :icpc - Intel Cilk Plus Compiler is used
 *
 *                           
 ************************************************************/

#include<iostream>
#include<stdlib.h>
#include <sys/time.h>
#include<errno.h>
#include <cilk/cilk.h>
#include <cilk/cilk_api.h>
#include<ctype.h>


using namespace std;

int vector_size;

double wallTime()
{
        double tsec=0.0;
        struct timeval mytime;
        gettimeofday(&mytime,0);
        tsec=(double)(mytime.tv_sec+mytime.tv_usec*1.0e-6);
        return tsec;
} 

double *dvector( int vector_size);


// allocates memory for an array of doubles , 
// return zero if falure 

double *dvector( int vector_size)
	{
		double *a;
		if(vector_size <= 0)
		{ 
			cout<< "illegal range of vector_size" << vector_size << endl;
			return 0;
		}
		a = new double[vector_size]; // allocates the memory for vector_size
		if(!a)
		{
			cout<< "memory allocation failed\n" << endl;
			return 0;
		}
	return a;
	}
			
       
void freedvector( double *a )
{
	delete [] &(a[0]);
}

int main(int argc , char *argv[])
{
	double sus_perf,user_perf;
        double t_perf;
	double *vector_a;
	double *vector_b;
	double *vector_c;
	//struct timeval start,end;
	double start,end;
        int rc ;
	double sum=0.0   ;


	if(argc < 2)
		{
			cout << "syntax <Vector_size>\n" << endl;
			return -1;
		}
		 vector_size=atoi(argv[1]);

	vector_a = dvector(vector_size) ; // allocates the memory for vector a
	if(!vector_a)
	{

		cout << "memory allocation is failed" << endl;
		return -1;
	}

	vector_b = dvector(vector_size) ; // allocates the memory for vector b
	if(!vector_b)
	{
		cout << "memory allocation is failed " << endl;
		return -1;
	}

	vector_c = dvector(vector_size);  // allocates the memory for vector c
	if(!vector_c)
	{
		cout << "memory allocation is failed " << endl;
		return -1;
	}

// Setting  elements of an array to a value array using Array
//     Notation
        
         vector_a[0:vector_size] = 5.0F;
	 vector_b[0:vector_size] = 4.0F;

       /*  for ( int n = 0 ; n < vector_size ; n ++ ) {
    cout << "a[" << vector_a[n] << "]:\n " ;
    cout << "b[" << vector_b[n] << "]:\n " ;
    
  }*/

	start=wallTime();                                                   

         vector_c[0:vector_size] = vector_a[0:vector_size] * vector_b[0:vector_size] ;
                        

                      // sum = sum + vector_c[0:vector_size];

		/*cilk_for (int  i = 0 ; i < vector_size ; i++)
			 {
				 //sum +=  vector_a[i] * vector_b[i];
				 sum +=  vector_c[i] ;
			 } 
*/


    //  mysum += vector_a[0:vector_size] * vector_b[0:vector_size] ;

			       //   sum += mysum;	

                            cout << "value of sum = " <<__sec_reduce_add(vector_c[0:vector_size]) << endl; 	

	end=wallTime();
  /*     for ( int n = 0 ; n < vector_size ; n ++ ) {
    	cout << "c[" << vector_c[n] << "]:\n " ;

  	}  */


        user_perf=((((double)vector_size)*2))/(end-start);



        /* cout << "-----------------------------------------------------------------------------------\n" << endl;

        cout <<"size of vector                              " << vector_size << endl;
        cout <<"size of data                                " << sizeof(double) << endl;
        cout <<"time elapsed                                " << end-start << endl;
        cout <<"Floating point operations                FLOPS " << (user_perf) << endl;

        cout <<"-----------------------------------------------------------------------------------\n" << endl;

*/
     
        cout <<"time elapsed using CILK                         " << end-start << endl;

	// Sequential
	
	sum=0.0;
	start=wallTime();

	for (int  i = 0 ; i < vector_size ; i++)
	 {
				 sum +=  vector_a[i]*vector_b[i];
	 } 
	end=wallTime();
        cout <<"time elapsed using sequential                         " << end-start << endl;
        cout << "value of sum = " << sum << endl; 	


   freedvector( vector_a  ) ;
   freedvector( vector_b  ) ;
   freedvector ( vector_c  ); 

  return 0;

} 
