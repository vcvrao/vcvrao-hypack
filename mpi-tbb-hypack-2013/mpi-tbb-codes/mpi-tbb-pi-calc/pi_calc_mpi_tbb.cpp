/*        C-DAC Tech Workshop : hyPACK-2013 
 *                  August 2013 
 *                    
 *
 * File	  : pi_calc.cpp
 * Desc   : to calculate the value of pi using mpi and  tbb
 * Input  : <Number of intervals> <grain_size>
 * Output : Successful/error , pi value and 
 *          the error in the calculation of the pi value.
 */

#include<iostream>
#include <mpi.h>// for using mpi fns
#include <stdlib.h>
#include <math.h>//the std math library
#include<tbb/task_scheduler_init.h>//to init the tbb task scheduler
#include<tbb/parallel_reduce.h>//for tbb's parallel reduce 
#include<tbb/blocked_range.h>
#include<sys/time.h> //for gettimeofday()
#include<error.h>

using namespace std;
using namespace tbb;

//Class for calculating a portion of pi using parallel_reduce
class pi_calc
{
  public:
  int num_intervals;
  double h,sum,x_tmp;
  double mypi;

  double func(double x)
  {
        return (4.0 / (1.0 + x*x));
  }

  void operator()(const blocked_range<size_t> &r)
  {
    for(size_t i=r.begin();i!=r.end();++i)
    {
      x_tmp=h*((double)i-0.5);
      sum+=func(x_tmp);
    }
  }
  

  pi_calc(pi_calc &x,split):
   num_intervals(x.num_intervals),//initialize num_intervals to n_intervals passed 
   h(1/(double)x.num_intervals),//initialize h to 1/num_intervals
   sum(0.0),//initialize sum to 0 
   x_tmp(0.0), //initialize x_tmp to 0
   mypi(0.0) //initialize mypi to 0
  {}

  void join(const pi_calc &y)
  {
    sum+=y.sum;
  }

  pi_calc(int n_intervals):
     num_intervals(n_intervals),//initialize num_intervals to n_intervals passed 
     h(1/(double)n_intervals),//initialize h to 1/num_intervals
     sum(0.0),//initialize sum to 0 
     x_tmp(0.0), //initialize x_tmp to 0
     mypi(0.0) //initialize mypi to 0
     {}

  void set_mypi()
  {
    mypi=h*sum;
  }
};

void terminate(int terminate_flag) // for debug purpose
{
 if(terminate_flag==1)
 {
   MPI_Finalize();
   exit(1);
 }
}

int main(int argc,char * argv[])
{
  int i;
  int my_rank;//to store own rank
  int num_procs;//num. of proc in the mpi universe
  int source,source_tag;
  int destination,destination_tag;
  int num_intervals;//intervals for integration (user input)
  int each_work;//the amount of work each process has to do
  MPI_Status status;//for mpi recv
  double PI_actual=3.141592653589793238462643; //actual pi val. upto 25dgt to compute error
  double pi=0,mypi=0;
  struct timeval tv_start,tv_end; //for gettimeofday
  struct timezone tz_start,tz_end;//for gettimeofday
  double timetaken;//to store the total time taken  
  int terminate_flag=0;//debug purpose
  
  MPI_Init(&argc,&argv); //initializing the mpi lib
  MPI_Comm_size(MPI_COMM_WORLD,&num_procs);//querying for no. of procs in the universe
  MPI_Comm_rank(MPI_COMM_WORLD,&my_rank);//querying for own rank
  int grain_size=1000;//grain_size for tbb parallel_reduce
 
 //if root then take input and send to all
 if(my_rank==0) 
  {
    cout<<"Number of intervals :";
    cin>>num_intervals;
    cout<<"Grain size at node level :";
    cin>>grain_size;

    for(i=1;i<num_procs;i++)
    {
      destination=i;
      destination_tag=0;
      MPI_Send(&num_intervals,1,MPI_INT,destination,destination_tag,MPI_COMM_WORLD);
      MPI_Send(&grain_size,1,MPI_INT,destination,destination_tag,MPI_COMM_WORLD);
    }
    
  }//otherwise recieve the inputs
  else
  {
    source=0;
    source_tag=0;
    MPI_Recv(&num_intervals,1,MPI_INT,source,source_tag,MPI_COMM_WORLD,&status);
    MPI_Recv(&grain_size,1,MPI_INT,source,source_tag,MPI_COMM_WORLD,&status);
  }

  terminate_flag=0;
  terminate(terminate_flag);//debug purpose
  
  if(num_intervals<=0) // error check
  {
    if(my_rank==0)
    {
      cout<<"invalid input for the number of intervals"<<endl;
    }

    MPI_Finalize();
    exit(1);
  }

// calculating the work to be done by each proc
 if(num_intervals%num_procs==0)
 {
     each_work=num_intervals/num_procs;
 }
 else
 {  
    cout<<"error: the no of intervals should be a multiple of the no of processes"<<endl;
    MPI_Finalize();
    exit(1);
 }
 
 //initializing the tbb task scheduler
 task_scheduler_init init;
 pi_calc pcl(num_intervals); // declaring object of class pi_calc

 if(my_rank==0)
 {
  //start time
  gettimeofday (&tv_start, &tz_start);
 }
 
 //invoking the parallel_reduce operation using the decided grainsize
  parallel_reduce(blocked_range<size_t>(my_rank*each_work,(my_rank*each_work)+each_work,grain_size), pcl);
// getting the calculated val of pi portion to the mypi variable of the object pcl 
  pcl.set_mypi();

  terminate_flag=0; 
  terminate(terminate_flag);//debug purpose
   
 //if root then recv the mypi vals from all other procs and calc the total pi val.  
 if(my_rank==0)
 {
   pi+=pcl.mypi;
   for(i=1;i<num_procs;i++)
   {
     source=i;
     source_tag=0;
     MPI_Recv(&mypi,1,MPI_DOUBLE,source,source_tag,MPI_COMM_WORLD,&status);
     pi+=mypi;
   }
 }
 else// otherwise send the mypi val to the root proc (rank 0)
 {
   destination=0;
   destination_tag=0;
   MPI_Send(&pcl.mypi,1,MPI_DOUBLE,destination,destination_tag,MPI_COMM_WORLD);
 }


 //if root then calc time taken , display pi val and the error in the calculation 
if(my_rank==0)
 {
  //end time
  gettimeofday(&tv_end,&tz_end);
  printf("pi is approximately %.16f, Error is %.16f\n",pi, fabs(pi - PI_actual));
  timetaken = tv_end.tv_sec * 1000000 + tv_end.tv_usec - (tv_start.tv_sec * 1000000 +	tv_start.tv_usec);
  cout << "Time taken (seconds) :" << timetaken / 1000000 << endl;//converting microseconds into seconds
  cout << "Time taken (u seconds)   :" << timetaken << endl;//time taken in microseconds
 }

 MPI_Finalize(); //all proc. now call MPI_Finalize
  return 0;
}
