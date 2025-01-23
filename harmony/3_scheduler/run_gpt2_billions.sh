#!/bin/bash
MODEL_DIR="../results"
for MODEL in "gpt2_2-5b" "gpt2_5b" "gpt2_10b" "gpt2_12-5b" "gpt2_15b" # "gpt2_12-5b" "gpt2_15b" # "gpt2_2-5b" # "gpt2_5b" "gpt2_7-5b" "gpt2_10b" "gpt2_12-5b" "gpt2_15b"
# do
  
#   # -------------- Manual ---------------------------
#   for D in 32
#   do
#     for MODE in "vPP"
#     do
#       for N in 4
#       do
#       echo "Manual"
#       python3 scheduler.py \
#       --manual \
#       --manual_ufwd 1 \
#       --manual_ubwd 1 \
#       --manual_packsize 2 \
#       --module_dir ${MODEL_DIR} \
#       --module_name ${MODEL} \
#       --minibatchsize ${D} \
#       --mode ${MODE} \
#       --num_gpus ${N} \
#       --simulation
#       done
#     done
#   done

# done

do

BILLION=$(echo ${MODEL} | sed 's/gpt2_//g')
echo "running Model: ${MODEL}, Billion: ${BILLION}"

for D in 16 32 64 128 256
do
  for MODE in 'vPP'
  do
    for N in 4
    do
    echo "Search"
    python3 scheduler.py \
    --packing_method_fwd 'balanced_time' 'reuse' \
    --packing_method_bwd 'balanced_time' \
    --topk 1 \
    --rank_fit_normally \
    --module_dir ${MODEL_DIR} \
    --module_name ${MODEL} \
    --minibatchsize ${D} \
    --mode ${MODE} \
    --num_gpus ${N} \
    --suffix "_gpt2_${BILLION}_3090" \
    done
  done
done

done