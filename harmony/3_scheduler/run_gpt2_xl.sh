#!/bin/bash
MODEL_DIR="../results"
MODEL="gpt2_xl"
# -------------- Manual ---------------------------
# for D in 32 256
# for D in 32
# do
#   # for MODE in "vDP" "vPP"
#   for MODE in "vPP"
#   do
#     for N in 4
#     do
#     echo "Manual"
#     python3 scheduler.py \
#     --manual \
#     --manual_ufwd 2 \
#     --manual_ubwd 2 \
#     --manual_packsize 7 \
#     --module_dir ${MODEL_DIR} \
#     --module_name ${MODEL} \
#     --minibatchsize ${D} \
#     --mode ${MODE} \
#     --num_gpus ${N} \
#     --simulation \
#     --verbose
#     done
#   done
# done
# -------------- Search ---------------------------
# for D in 32 256
# do
#   for MODE in 'vDP' 'vPP'
#   do
#     for N in 1 2 4
#     do
#     echo "Search"
#     python3 scheduler.py \
#     --packing_method_fwd 'balanced_time' 'reuse' \
#     --packing_method_bwd 'balanced_time' \
#     --topk 1 \
#     --rank_fit_normally \
#     --module_dir ${MODEL_DIR} \
#     --module_name ${MODEL} \
#     --minibatchsize ${D} \
#     --mode ${MODE} \
#     --num_gpus ${N}
#     done
#   done
# done
for D in 16 32 64 128 256
do
  for MODE in 'vDP' 'vPP'
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
    --suffix "_gpt2_xl_3090"
    # --verbose \
    # |& tee gpt2_medium_原版结果.txt
    done
  done
done