#
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
#
#uncomment for execution on nvidia-cluster 
#CC = /state/partition1/apps/pgi_12_3/linux86-64/12.3/bin/pgcc
#CCFLAGS = -I/state/partition1/apps/pgi_12_3/linux86-64/12.3/include
#ACCFLAGS = -acc -ta=nvidia -Minfo=accel -lpgacc
#OMPFLAGS = -fast -fopenmp -Minfo

CC = pgcc

#test1 for openacc and test2 is for sequential execution
#uncomment the object which you want to execute
object = vect-vect-openacc
#object = vect-vect-sequential


run: 
	$(CC) $(CCFLAGS) $(ACCFLAGS) -o $(object) $(object).c


clean:
	rm -r  $(object)
