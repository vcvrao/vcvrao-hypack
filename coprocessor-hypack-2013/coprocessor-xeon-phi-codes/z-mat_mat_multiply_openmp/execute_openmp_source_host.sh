#
#	C-DAC Tech Workshop : hyPACK-2013
#             October 15-18, 2013
#
#   Created           :  August-2013
#
#   E-mail            :  hpcfte@cdac.in     
#
# Square Matrix Size (Size) = 256
# Max Iterations (Max) = 10 
# No. of Threads used in OpenMP (i) = 1

#!/bin/bash

i=1
Max=5
Size=1024
while test "$i" -lt $Max ;do
  ./run $Size $i 2 
 echo "$i"
let i++
#i = $((i + 1))
done

  
