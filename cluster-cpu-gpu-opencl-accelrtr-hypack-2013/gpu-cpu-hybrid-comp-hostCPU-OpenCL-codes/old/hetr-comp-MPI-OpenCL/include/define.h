/*#*****************************************************************************************
#
#                          C-DAC Tech Workshop : HEMPA-2011
#                             Oct 17 - 21, 2011
#
#  Created         : Aug 2011 
#
#  Email           : betatest@cdac.in        
#****************************************************************************************/

/* common pragma defination that are goin to be used through out the OpenCL programs */
#define GROUP_SIZE 4
#define NUMTHREAD  4
#define STATUSCHKMSG(x) if(status != CL_SUCCESS) { cout<<"\n Operation is not successful : ";cout<<x<<"\n"; exit(1);}
#define STOP    cout<<"program terminated here";exit(-1);
#define BUFFER_SIZE 100                /* size to define temprary buffer size , which hold process, platform, device name */ 
#define MAX_NUM_OF_PLATFORM_AND_DEVICE_ON_EACH_NODE 10
#define BUILD_USER_DEF_KERNEL	1
#define BUILD_LIBR_DEF_KERNEL	0
