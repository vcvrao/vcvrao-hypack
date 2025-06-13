#*****************************************************************************************
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
# Created         : August 2013 
#
# Email           : hpcfte@cdac.in        
#*****************************************************************************************/
#
##### User configurable options #####
# Bellow are the configuration and installtion detail of MPI-OpenCL src package
# named as "heterogeneous-computing". All the bellow path and software installation
# details need to be set by user. Kindly read detail instruction for each section 
# bellow.

#heterogenious-computing config
# MPIOCLHOME is the absulute path to the dir "heterogeneous-computing" package's.
MPIOCLHOME=/home/hempa101/wk-samritm/hetr-comp-MPI-OpenCL
MPIOCLINC=$(MPIOCLHOME)/include
MPIOCLSRC=$(MPIOCLHOME)/src
MPIOCLBIN=$(MPIOCLHOME)/bin



## MPI configuration
## bellow are the detail of MPI library installation detail on the targated system.
CC = mpicxx
LIBS = -lm 
CFLAGS = -DMPICH_IGNORE_CXX_SEEK   
LFLAGS = -lpthread -ldl  

## AMD APP SDK configuration 
# AMD APP SDK  installation detail on targated system
CLHOME=/usr/local/AMD-APP-SDK-v2.4-lnx64
CLINCLUDE=$(CLHOME)/include
CLLIB=$(CLHOME)/lib/x86_64
CLLINKFLAGS=-lOpenCL

#AMD CLAMDBLAS configuration
# AMD OPENCL BLAS installation detail.
CLAMDBLASHOME=/usr/local/clAmdBlas-1.4
CLAMDBLASINC=$(CLAMDBLASHOME)/include
CLAMDBLASLIB=$(CLAMDBLASHOME)/lib64
CLAMDBLASFLAG= -lclAmdBlas


