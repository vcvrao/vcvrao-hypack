#---------------------------------------------------------------------------
# Items in this file will need to be changed for each platform.
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# Parallel C:
#
# For IS, which is in C, the following must be defined:
#
# CC         - C compiler 
# CFLAGS     - C compilation arguments
# C_INC      - any -I arguments required for compiling C 
# CLINK      - C linker
# CLINKFLAGS - C linker flags
# C_LIB      - any -L and -l arguments required for linking C 
#
# compilations are done with $(CC) $(C_INC) $(CFLAGS) or
#                            $(CC) $(CFLAGS)
# linking is done with       $(CLINK) $(C_LIB) $(CLINKFLAGS)
#---------------------------------------------------------------------------

#---------------------------------------------------------------------------
# This is the C compiler used for C programs
#---------------------------------------------------------------------------
CC = gcc
# This links C programs; usually the same as ${CC}
CLINK	= $(CC)

#---------------------------------------------------------------------------
# These macros are passed to the linker 
#---------------------------------------------------------------------------
C_LIB  = -lpthread -lm

#---------------------------------------------------------------------------
# These macros are passed to the compiler 
#---------------------------------------------------------------------------
C_INC =

#---------------------------------------------------------------------------
# Global *compile time* flags for C programs
# DC inspects the following flags (preceded by "-D"):
#
# IN_CORE - computes all views and checksums in main memory (if there is 
# enough memory)
#
# VIEW_FILE_OUTPUT - forces DC to write the generated views to disk
#
# OPTIMIZATION - turns on some nonstandard DC optimizations
#
# _FILE_OFFSET_BITS=64 
# _LARGEFILE64_SOURCE - are standard compiler flags which allow to work with 
# files larger than 2GB.
#---------------------------------------------------------------------------
CFLAGS	= -O3 
#---------------------------------------------------------------------------
# Global *link time* flags. Flags for increasing maximum executable 
# size usually go here. 
#---------------------------------------------------------------------------
CLINKFLAGS = -O3 -lpthread -lm 


#---------------------------------------------------------------------------
# Utilities C:
#
# This is the C compiler used to compile C utilities.  Flags required by 
# this compiler go here also; typically there are few flags required; hence 
# there are no separate macros provided for such flags.
#---------------------------------------------------------------------------
UCC	= gcc


#---------------------------------------------------------------------------
# Destination of executables, relative to subdirs of the main directory. . 
#---------------------------------------------------------------------------
BINDIR	= ../bin





