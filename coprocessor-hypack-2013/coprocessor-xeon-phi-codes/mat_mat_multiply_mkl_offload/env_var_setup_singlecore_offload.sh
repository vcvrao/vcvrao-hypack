#
#               C-DAC Tech Workshop : hyPACK-2013
#                        October 15-18, 2013
#
#   Created           :  August-2013
#
#   E-mail            :  hpcfte@cdac.in
#
#!/bin/bash


# compilation of source code
make -f Makefile_MKL_SingleCore.OFFLOAD clean
make -f Makefile_MKL_SingleCore.OFFLOAD

#execution of source code
./run 0 0 16000

make -f Makefile_MKL_SingleCore.OFFLOAD clean
