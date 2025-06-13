#
# *******************************************************************
#
#		C-DAC Tech Workshop : hyPACK-2013
#                           October 15-18, 2013
#
# *******************************************************************
#   Created             : August 2013
#
#   E-mail              : hpcfte@cdac.in     
#
#uncomment for compilation on Nvidia cluster 
#CC       = /state/partition1/apps/pgi_12_3/linux86-64/12.3/bin/pgcc
#CCFLAGS  = -I/state/partition1/apps/pgi_12_3/linux86-64/12.3/include
#LIB = -L/state/partition1/apps/pgi_12_3/linux86-64/12.3/lib 


#compilation on leopard
CC       = pgcc

#pgcc flags
#ACCFLAGS = -acc -ta=nvidia,time -Minfo=accel
#ACCFLAGS =  -acc -ta=nvidia,time -Minfo=accel
#OMPFLAGS = -fast -mp -Minfo

object = test-MatMult.c

run:
	#$(CC) -acc -ta=nvidia -fast -Minfo -o test $(object)
	$(CC) -acc -ta=nvidia,time  -o test $(object)
clean:	
	$(RM) test 
 
