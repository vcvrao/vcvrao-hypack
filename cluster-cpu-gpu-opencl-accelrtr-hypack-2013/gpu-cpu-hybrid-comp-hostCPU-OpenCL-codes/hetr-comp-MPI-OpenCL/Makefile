#
#*******************************************************************
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
#  Created         : August 2013 
#
#  Email           : hpcfte@cdac.in        
#
#***********************************************************************
#
##### User configurable options #####
include make.inc

TARGETS= hybridPrefixSum

run:$(TARGETS)

hybridPrefixSum:
	-(make -f ./src/$@/Makefile clean)    
	-(make -f ./src/$@/Makefile)    

