        Center for Development of Advanced Computing (C-DAC):  December-2006
	C-DAC Multi Core Benchmark Suite 1.0 
        Shared Memory Programming - POSIX Threads(Pthreads) 
	Email : betatest@cdac.in
  ------------------------------------------------------------------------------------
   
   Table of Contents
   [1]Introduction
   [2]Compiling
   [3]Running
   [4]Analyzing the Output

   Introduction:
        
   This is  benchmark comprises of suites performing Integer / Floating-Point
   Numerical and Non-Numerical computations using Shared Memory Programming (PThreads) .
   These suites measures the execution time of kernels of Dense Matrix Computations 
   involving Computation of Square Matrix Norm by Rowwise/Columnwise Partitioning ,Matrix
   and Vector Multiplication using checkerboard algorithm & Matrix and Matrix Multiplication 
   using self scheduling algorithm ; PI computation using Numerical Integration and 
   Monte Carlo Method ; Solving the Linear equation Ax = b using Jacobi Method ; and
   Sorting the given single dimension array for finding the minimum integer. 
 
   The suites run for problem sizes - Class A,B,C on 1/2/4/8 threads.

                This Multi Core Benchmark gives the performance of system in terms of
   Time , Memory Utilized, Cpu Speed etc.


   Compiling:

   Before compilation, prepare configuration file 'Make.inc' in the config
   directory with the help of a templete provided.
   Use the following command to compile the benchmark.


       make <benchmark-name> THREADS=<number> CLASS=<class> 

   where <benchmark-name>  is "pi", "jacobi", "int_sort", "mat_cmp_in", "mat_cmp_db", 
         <number>          is the number of threads to be spawned 
         <class>           is "A", "B", and "C"

   Running:

       The executable is named <benchmark-name>.<threads>.<class>
   The executable is placed in the bin subdirectory. 
   To run the benchmark use ./bin/benchmark-name>.<threads>.<class>   

   Analyzing the Output:

   Output for all these Suites will provide the 
   <1>. Status of the Suite.
   <2>. Number of Threads utilized for Suite.
   <3>. Amount of Memory utilized for performing the computation in terms of MB.
   <4>. Total amount of Time taken for performing the computation in terms of Seconds.
   <5>. Summary of Compiler and Libraries used.  
