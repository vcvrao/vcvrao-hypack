
/* All Global variable define here */

#ifndef _SYSHEADER_H
//  #define _SYSHEADER_H 1
  #include <sys/types.h>
#endif

#ifndef _DEFINE_H


#define MAP_RDONLY   1
#define MAP_RDWR     2


using namespace std;

extern int fda, fdb, fdc,fds;
extern size_t nrows, ncols;
extern float *ma, *mb, *mc,*ms;
extern size_t mapsize;

#endif
