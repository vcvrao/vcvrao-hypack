__kernel
void 
prefixSum_kernel(__global int *inArray, __global int *outArray, __global int* arrayLength)
 {
    unsigned int gid = get_global_id(0);
    unsigned int numDim = get_work_dim(); 
    unsigned int prefixSum = 0;
    unsigned int numThread = 1;
    for( int count=0; count< numDim;  count++){
	numThread = numThread * get_global_size(count);
    }
	
   for( int curCell = gid; curCell < (*arrayLength); curCell = curCell + numThread){
	prefixSum = 0;
	for( int count = 0; count <= curCell; count++){
		prefixSum += inArray[count];
	}
    outArray[curCell] = prefixSum ;
    }


 } //end of PrefixSum_kernel
