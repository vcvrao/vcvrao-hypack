#
#**********************************************************************
#
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
# Created               : August-2013
# 
#       
#*************************************************************************




export OFFLOAD_REPORT=2
export OFFLOAD_DEVICES=0,1
export OFFLOAD_WORKDIVISION=1

./run 1000000000 236

unset OFFLOAD_REPORT
unset OFFLOAD_DEVICES
unset OFFLOAD_WORKDIVISION
