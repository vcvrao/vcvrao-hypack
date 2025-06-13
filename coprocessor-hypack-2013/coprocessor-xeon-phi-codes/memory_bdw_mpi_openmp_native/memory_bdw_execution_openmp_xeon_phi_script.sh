#
#    C-DAC Tech Workshop : hyPACK-2013
#           October 15-18, 2013
#
#   Created    :  August-2013
#
#   E-mail     :  hpcfte@cdac.in     
#
#!/bin/bash

i=1
while test "$i" -lt 6 ;do
  export OMP_NUM_THREADS=236
 # echo $OMP_NUM_THREADS 
  echo "------------------------------------------" 
  #./run 1024 $i 10
  ./run
let i=$((i * 2))
#i=$((i + 1))
done

