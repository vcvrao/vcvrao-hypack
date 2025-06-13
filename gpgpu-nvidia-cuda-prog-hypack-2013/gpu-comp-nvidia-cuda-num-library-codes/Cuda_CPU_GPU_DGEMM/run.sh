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
#!/bin/bash

# Shell script to compile and execute the program for different
# data sizes in single stroke and generate the result/log in
# ./result directory

HOSTNAME=`hostname`
DATE=`date +%d-%m-%Y-%H_%M_%S`
RESULT_FILE=./results/result-dgemm_$DATE.txt
COMPILE_LOG_FILE=./results/compile-dgemm-log_$DATE.txt


## If one want to run for another data size rather than mentioned below,
## please add that data size into DATA_SIZE variable
DATA_SIZE=("1024" "2048" "4096" "5192" "8192" "10240" "12288")

CODE_NAME=("./bin/cpu-gpu-dgemm" ) 

PRINT_CODE_NAME=(" CPU_GPU_DGEMM")


echo					> $RESULT_FILE 
echo " Executing on HOST : $HOSTNAME"   >>$RESULT_FILE
echo " Date of Execution : $DATE" 	>>$RESULT_FILE
echo>>$RESULT_FILE

echo 
echo "Compiling Codes ......... "	
echo "Compiling Codes ......... "	>>$RESULT_FILE
make clean				> $COMPILE_LOG_FILE 
make compile_all 			>> $COMPILE_LOG_FILE
if [ $? -ne 0 ]
then
	echo " Compilation failed. Please refer the compilation log in ./results directory" 	>>$RESULT_FILE
	echo " Compilation failed. Please refer the compilation log in ./results directory" 	
	exit 1
fi
 
echo "Compilation Done ......... " 	>>$RESULT_FILE
echo 					>>$RESULT_FILE
echo "Compilation Done ......... " 
echo 					>>$RESULT_FILE
echo "Exection started !!"

echo " Executing Codes  "		>>$RESULT_FILE
echo "--------------------------------------------------------------------------------">>$RESULT_FILE



for (( i=0; i<${#CODE_NAME[@]}; i++)); 
do
	echo										>>$RESULT_FILE
	echo " ************** Executing : ${PRINT_CODE_NAME[$i]} ********************* ">>$RESULT_FILE
	echo>>$RESULT_FILE
	echo										 >>$RESULT_FILE

	for (( j=0; j<${#DATA_SIZE[@]}; j++)); 
	do
		${CODE_NAME[$i]} ${DATA_SIZE[$j]} ${DATA_SIZE[$j]} ${DATA_SIZE[$j]} 	>>$RESULT_FILE
		if [ $? -ne 0 ]
		then
        		echo " Execution failed. Please refer the  ./results directory"     >>$RESULT_FILE
       			echo " Execution failed. Please refer the  ./results directory"     
        		exit 1
		fi

	done
done

echo "--------------------------------------------------------------------------------" >>$RESULT_FILE
echo " Executing Codes : Successful  "							>>$RESULT_FILE
echo " Executing Codes : Successful  "						
echo " Please refer the complilation log and result file in ./results directory "
echo 
echo					

