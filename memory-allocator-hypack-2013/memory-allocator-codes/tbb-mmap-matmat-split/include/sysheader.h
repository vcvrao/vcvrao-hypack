
/* Header File Include Here */
#ifndef _SYSHEADER_H
  #define _SYSHEADER_H 1
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>
#include <assert.h>
#include <iostream>

/*TBB related header file include here */

#include"tbb/task_scheduler_init.h"
#include<tbb/parallel_for.h>
#include<tbb/blocked_range.h>
#include<tbb/blocked_range2d.h>
#include<tbb/tick_count.h>

#endif


