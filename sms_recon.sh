#!/bin/bash
# Script to do SMS reconstructions
# Author: Tarun Naren

#export PATH="$HOME/local/bin:$PATH"
#export PATH="$HOME/local/scripts:$PATH"

FILENAME=""
PCVIPR=""
SMS_FACTOR=2
RCFRAMES=40
RESP="exp"
CR="sense"
ITER="10"
EPOCH="20"
LAMBDA="0.00001"
TIME="full"
OUT_NAME="SMS"
ADD_FLAGS=()

usage() {
	echo 'Usage: Script to reconstruct all SMS slices
	-f [path/to/data]
	-p [do pcvipr_recon before SMS recon to export MRI_Raw file (provide path/to/raw_data)]
	-n [sms factor, default is 2]
	-c [# of cardiac frames, default is 40] 
	-r [respiratory type (exp, insp, or none, default is exp)]
	-l [constrained recon (pils, sense, llr, or mslr, default is pils)]
	-i [number of iterations for constrained recon, default is 10]
	-e [number of epochs for mslr recon, default is 20]
	-y [lambda value for sense, default is 0.01]
	-t [retrospective scan time control, default is using all data (specify in format start_time,end_time in seconds; Ex:0-120,230-400)]
	-o [output file name]
	-x [additional flags...]
	-h [show this help message]'
}

exit_fail() {
    usage
    exit 1
}

while getopts 'f:p:n:s:r:c:l:i:e:y:t:o:z:x:h' flag; do
	case "${flag}" in
		f) FILENAME="${OPTARG}" ;;
		p) PCVIPR="${OPTARG}" ;;
		n) SMS_FACTOR="${OPTARG}" ;;
		c) RCFRAMES="${OPTARG}" ;;
		r) RESP="${OPTARG}" ;;
		l) CR="${OPTARG}" ;;
		i) ITER="${OPTARG}" ;;
		e) EPOCH="${OPTARG}" ;;
		y) LAMBDA="${OPTARG}" ;;
		t) TIME="${OPTARG}" ;;
		o) OUT_NAME="${OPTARG}" ;;
		z) RECON_TYPE="${OPTARG}" ;;
		x) ADD_FLAGS=("${OPTARG}") ;;
		h) usage 
			exit 1 ;;
		:) echo "Error: -${OPTARG} requires an argument"
			exit_fail ;;
		*) exit_fail ;;
	esac
done

if [ -n "$PCVIPR" ]; then
	echo "Running: pcvipr_recon first..."
	pcvipr_recon -f "${PCVIPR}" -rcframes 1 -export_kdata -export_smaps -threads 64 -out_folder SMS_2DPC
	cd SMS_2DPC/dat || exit
fi

if [ -z "$FILENAME" ]; then
	echo "Error: -f [path/to/data] is required"
	exit_fail
fi

if [ "${RECON_TYPE}" == "test" ]; then
	BASE_FLAGS=(--smap_type lowres --coil_batch_size 40 --flow_processing --sms_factor "${SMS_FACTOR}")
else 
	BASE_FLAGS=(--autofov --gate_type ecg --smap_type lowres --coil_batch_size 10 --flow_processing --sms_factor "${SMS_FACTOR}")
fi


if [ "$RESP" == "exp" ]; then
	RESP_FLAG=(--resp_gate --resp_filter_window 5 --resp_sign 1)
elif [ "$RESP" == "insp" ]; then
	RESP_FLAG=(--resp_gate --resp_filter_window 5 --resp_sign -1)
elif [ "$RESP" == "none" ]; then
	RESP_FLAG=(--resp_gate --resp_efficiency 1)
else
	echo "invalid respiratory type"
	usage
	exit_fail
fi

if [ "$CR" == "pils" ]; then
	CR_FLAG=(--recon_type pils)
elif [ "$CR" == "sense" ]; then
	CR_FLAG=(--recon_type sense --max_iter $ITER --lamda $LAMBDA)
elif [ "$CR" == "wavelet" ]; then
	CR_FLAG=(--recon_type wavelet --max_iter $ITER --lamda $LAMBDA)
elif [ "$CR" == "llr" ]; then
	CR_FLAG=(--recon_type llr --max_iter $ITER --lamda $LAMBDA --llr_block_width 16)
elif [ "$CR" == "mslr" ]; then
	CR_FLAG=(--recon_type mslr --max_iter $ITER --lamda $LAMBDA --epochs $EPOCH)
else
	echo "invalid constrained recon type"
	usage
	exit_fail
fi

if [ "${TIME}" == "full" ]; then
	# IFS=',' read -ra time_params <<< "$TIME"
	TIME_FLAG=()
else
	TIME_FLAG=(--time_range "${TIME}")
fi

echo llr_recon_flow.py --filename "${FILENAME}" "${BASE_FLAGS[@]}" --frames "${RCFRAMES}" "${CR_FLAG[@]}" "${RESP_FLAG[@]}" "${TIME_FLAG[@]}" "${ADD_FLAGS[@]}" --out_filename "${OUT_NAME}"
llr_recon_flow.py --filename "${FILENAME}" "${BASE_FLAGS[@]}" --frames "${RCFRAMES}" "${CR_FLAG[@]}" "${RESP_FLAG[@]}" "${TIME_FLAG[@]}" "${ADD_FLAGS[@]}" --out_filename "${OUT_NAME}" | tee "pyrecon_${OUT_NAME}.log"
# if [ "${SLICE}" == "all" ]; then
# 	for i in $(seq 0 $((SMS_FACTOR-1))); do
# 		SMS_FLAG=(--sms_slice "${i}")
# 		NAME="${OUT_NAME}_slice${i}.h5"
# 		echo llr_recon_flow.py --filename "${FILENAME}" "${BASE_FLAGS[@]}" --frames "${RCFRAMES}" "${CR_FLAG[@]}" "${RESP_FLAG[@]}" "${SMS_FLAG[@]}" "${TIME_FLAG[@]}" "${ADD_FLAGS[@]}" --out_filename "${NAME}"
# 		llr_recon_flow.py --filename "${FILENAME}" "${BASE_FLAGS[@]}" --frames "${RCFRAMES}" "${CR_FLAG[@]}" "${RESP_FLAG[@]}" "${SMS_FLAG[@]}" "${TIME_FLAG[@]}" "${ADD_FLAGS[@]}" --out_filename "${NAME}" | tee pyrecon_${NAME}.log
# 	done
# else
# 	SMS_FLAG=(--sms_slice "${SLICE}")
# 	NAME="${OUT_NAME}_slice${SLICE}.h5"
# 	echo llr_recon_flow.py --filename "${FILENAME}" "${BASE_FLAGS[@]}" --frames "${RCFRAMES}" "${CR_FLAG[@]}" "${RESP_FLAG[@]}" "${SMS_FLAG[@]}" "${TIME_FLAG[@]}" "${ADD_FLAGS[@]}" --out_filename "${NAME}"
# 	llr_recon_flow.py --filename "${FILENAME}" "${BASE_FLAGS[@]}" --frames "${RCFRAMES}" "${CR_FLAG[@]}" "${RESP_FLAG[@]}" "${SMS_FLAG[@]}" "${TIME_FLAG[@]}" "${ADD_FLAGS[@]}" --out_filename "${NAME}" | tee pyrecon_${NAME}.log
# fi
