

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>                                  // header file inclusion
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <iostream>

#include "tbb/task_scheduler_init.h"
#include <tbb/parallel_for.h>			// tbb related header file inclusion
#include <tbb/blocked_range.h>
#include<tbb/tick_count.h>


using namespace tbb;
using namespace std;

