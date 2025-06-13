

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>                                      // including header files which are required in remainning code
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <iostream>

#include <tbb/task_scheduler_init.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>                          // tbb related header files
#include<tbb/tick_count.h>
#include<tbb/concurrent_queue.h>
#include <tbb/spin_mutex.h>
#include <list>


