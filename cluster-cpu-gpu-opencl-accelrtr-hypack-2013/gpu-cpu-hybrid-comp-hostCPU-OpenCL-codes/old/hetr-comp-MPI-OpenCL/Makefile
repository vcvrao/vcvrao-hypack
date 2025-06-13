#*****************************************************************************************
#
#                          C-DAC Tech Workshop : HEMPA-2011
#                             Oct 17 - 21, 2011
#
#  Created         : Aug 2011 
#
#  Email           : betatest@cdac.in        
#****************************************************************************************
##### User configurable options #####
include make.inc

TARGETS= hybridPrefixSum

run:$(TARGETS)

hybridPrefixSum:
	-(make -f ./src/$@/Makefile clean)    
	-(make -f ./src/$@/Makefile)    

