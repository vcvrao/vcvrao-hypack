/*#*****************************************************************************************
#
#                          C-DAC Tech Workshop : HEMPA-2011
#                             Oct 17 - 21, 2011
#
#  Created         : Aug 2011 
#
#  Email           : betatest@cdac.in        
#****************************************************************************************/

#include"mpi.h"
#include<CL/cl.h>
#include<common.h>
#include<define.h>

// structure encapsulating all the required parameter by setExeEnv routine.
typedef struct exeEnvParamL{
                /*platform related parameters*/
                cl_uint maxNumOfPlatforms;
                cl_platform_id *platforms;
                cl_uint numOfPlatforms;

                /*device related parameters*/
                cl_uint maxNumOfDevices;
                cl_device_id *devices;
                cl_uint numOfDevices;
                cl_device_type deviceType;

                /*context and program compilation related parametrs*/
                cl_context context;
                cl_command_queue queue;
                
		/* user defined kernel detail */
		cl_uint kernelBuildSpec;
		char pathToSrc[100];
                cl_program hProgram;

                /*current MPI process related parameter, used to schedule current process and work to
                  a specific available device on compute node */
                cl_uint myProcID;
}exeEnvParams;


// @brief               Subroutine to load OpenCL Kernel at runtime.
// @param[in]           Path to the kernel OpenCL kernel.
// @return              Return a character string representing OpenCL Kernel source code.
char* readKernelSource(char* path)
 {
   int srcLen;
   char* sProgramSource;
   ifstream srcFile;
   srcFile.open(path, ifstream::in);

   srcFile.seekg(0,ios::end);
   srcLen = srcFile.tellg();
   srcFile.seekg(0,ios::beg);

   sProgramSource = ( char*) malloc( srcLen * sizeof(char));
   srcFile.read(sProgramSource, srcLen);
   return sProgramSource;
 }// end of readKernelSource


// @brief       Subroutine to set environment for kernel compilation
// @param[in,out]  context      Handle to current execution context.
// @param[in,out]  numOfDevices       Hold device list length.
// @param[in,out]  devices      Handle to list of devices.
// @param[in,out]  queue        Handle to command queue to currently used context with specific device.
// @param[in,out]  hProgram     Handle to kernel source program.
// @param[in]      path         Relative path to kernel source code.
// @param[in]      deviceType   Targated device type.
// @return      On successful execution returns void or nothing.
// @brief       Subroutine to set environment for kernel compilation
// @param[in,out]  context      Handle to current execution context.
// @param[in,out]  numOfDevices       Hold device list length.
// @param[in,out]  devices      Handle to list of devices.
// @param[in,out]  queue        Handle to command queue to currently used context with specific device.
// @param[in,out]  hProgram     Handle to kernel source program.
// @param[in]      path         Relative path to kernel source code.
// @param[in]      deviceType   Targated device type.
// @return      On successful execution returns void or nothing.
void setExeEnv( exeEnvParams *exeEnvParamList)
 {
   cl_int status = CL_SUCCESS;


   // basic initialization of exeEnvParamList variables
   exeEnvParamList->platforms = (cl_platform_id*) malloc (exeEnvParamList->maxNumOfPlatforms * sizeof(cl_platform_id));
   exeEnvParamList->devices = (cl_device_id*) malloc (exeEnvParamList->maxNumOfDevices * sizeof(cl_device_id));
   // get all available platform IDs.
   status = clGetPlatformIDs(exeEnvParamList->maxNumOfPlatforms, exeEnvParamList->platforms, &(exeEnvParamList->numOfPlatforms));
   STATUSCHKMSG("clGetPlatformIDs Failed ");
   // from a list of OpenCL capable platforms select one GPU device.
   //cout<<" Available OpenCL Platforms : \n";
   unsigned platformCount = 0;
   unsigned devCount = 0;
   cl_uint deviceFound = CL_FALSE;
   cl_uint platItrFlag = CL_TRUE;
   cl_uint startItrPlat = (exeEnvParamList->myProcID) % ( exeEnvParamList->numOfPlatforms);

   for( platformCount = startItrPlat; (platformCount != startItrPlat) | platItrFlag ; platformCount = (platformCount+1)%(exeEnvParamList->numOfPlatforms)){
        platItrFlag = CL_FALSE;
        status = clGetDeviceIDs( exeEnvParamList->platforms[platformCount], exeEnvParamList->deviceType, exeEnvParamList->maxNumOfDevices, exeEnvParamList->devices,&(exeEnvParamList->numOfDevices));
        // select every device available on system one by one 
        cl_uint devItrFlag = CL_TRUE;
        cl_uint startItrDev = (exeEnvParamList->myProcID) % (exeEnvParamList->numOfDevices);
        for( devCount = startItrDev; (devCount != startItrDev) | devItrFlag ; devCount = (devCount + 1) % (exeEnvParamList->numOfDevices) ){
                devItrFlag = CL_FALSE;
                cl_bool isDeviceAvail;
                size_t paramValueSizeRet;
                //cl_device_type        deviceType;
                status = clGetDeviceInfo((exeEnvParamList->devices)[devCount],CL_DEVICE_AVAILABLE,sizeof(cl_bool),&isDeviceAvail,&paramValueSizeRet);
                STATUSCHKMSG(status);
                //status = clGetDeviceInfo((exeEnvParamList->devices)[devCount],CL_DEVICE_TYPE,sizeof(cl_device_type),&deviceType,NULL);
                //STATUSCHKMSG(status);
                if( isDeviceAvail & CL_DEVICE_AVAILABLE) {
                        cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)exeEnvParamList->platforms[platformCount],0};
                        // create context type for given device type
                        cl_context_properties* cprops = ( NULL == exeEnvParamList->platforms[platformCount] ) ? NULL : cps;
                        cl_uint numOfDevWithContext = 1;
                        exeEnvParamList->context = clCreateContext(cprops, numOfDevWithContext, ((exeEnvParamList->devices)+devCount), NULL, NULL, &status);
                        STATUSCHKMSG(" context ");

                        // create command queue
                        exeEnvParamList->queue = clCreateCommandQueue( exeEnvParamList->context, (exeEnvParamList->devices)[devCount], 0, &status);
                        STATUSCHKMSG("command queue");
			
			// build specification : if user is going to define kernel , bellow section will be executed otherwise
			// if library kernel is going to be used then bellow section will be skiped. This depend on flag 
			// exeEnvParamList->kernelBuildSpec
			if( exeEnvParamList->kernelBuildSpec == BUILD_USER_DEF_KERNEL ) {
				// create a CL program using kernel source
                        	const char* sProgramSource = readKernelSource(exeEnvParamList->pathToSrc);
                        	size_t sourceSize[] = { strlen(sProgramSource) };
                        	exeEnvParamList->hProgram = clCreateProgramWithSource(exeEnvParamList->context, 1, &sProgramSource, sourceSize, &status);
                        	STATUSCHKMSG("create source handle");

                        	// built the program
                        	status = clBuildProgram( exeEnvParamList->hProgram,numOfDevWithContext, ((exeEnvParamList->devices)+devCount),NULL,NULL,NULL);
                        	STATUSCHKMSG("build");
			} //end of if
                        deviceFound = CL_TRUE;
                        break;
                }// end of if
        }// end of for  
        if( deviceFound ){break;}
    }// end of for
 }// end of setExeEnv


