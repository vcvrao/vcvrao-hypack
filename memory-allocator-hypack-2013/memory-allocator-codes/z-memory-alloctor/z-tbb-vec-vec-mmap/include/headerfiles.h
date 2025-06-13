

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>                         // header file inclusion
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <iostream>

#include "tbb/task_scheduler_init.h"
#include <tbb/parallel_for.h>                       // tbb related header files
#include <tbb/blocked_range.h>
#include<tbb/tick_count.h>


using namespace tbb;
using namespace std;
