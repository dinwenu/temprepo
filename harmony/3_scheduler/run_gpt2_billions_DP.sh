#!/bin/bash
MODEL_DIR="../results"
for MODEL in "gpt2_10b" # "gpt2_12-5b" "gpt2_15b" # "gpt2_2-5b" # "gpt2_5b" "gpt2_7-5b" "gpt2_10b" "gpt2_12-5b" "gpt2_15b"

do
# 提取模型大小并存入 BILLION 变量
BILLION=$(echo ${MODEL} | sed 's/gpt2_//g')
echo "正在处理 Model: ${MODEL}, Billion: ${BILLION}"

for D in 16 32 64 128 256
do
  for MODE in 'vDP'
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
    # |& tee gpt2_10b_结果.txt
    # --verbose \
    # |& tee gpt2_medium_原版结果.txt
    done
  done
done

done