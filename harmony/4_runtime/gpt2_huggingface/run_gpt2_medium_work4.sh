#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_DIR="../../results"
MODEL="gpt2_medium"
CONFIG="../../../model_lib/gpt2_configs/gpt2-medium-config.json"
INIT_MODEL="../../../pretrained_models/GPT2-Medium_Seed42"
SEED=42
# -------------------- Train ------------------------
# for SCHEDULED in "D32_vPP_N4_scheduler2_Top1" # "D32_vDP_N1_Ufwd2_Ubwd2_P7" "D32_vDP_N2_Ufwd2_Ubwd2_P7" "D32_vPP_N2_Ufwd2_Ubwd2_P7"
for SCHEDULED in "D32_vPP_N4_Ufwd8_Ubwd4_numP5" # "D32_vDP_N1_Ufwd2_Ubwd2_P7" "D32_vDP_N2_Ufwd2_Ubwd2_P7" "D32_vPP_N2_Ufwd2_Ubwd2_P7"
do
echo "Clean Python Processes"
# sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

OUT_DIR="./logs/finetune_${MODEL}/${SCHEDULED}"
mkdir -p ${OUT_DIR}
echo "${SCHEDULED}"
# nsys profile -o gpt2_medium_D32_vPP_N4_Top1 \
numactl --cpunodebind=0 --membind=0 \
python3 main_work4.py \
--gpt2_train_file "../../../data/wikitext-103-tokens/wiki.train.tokens" \
--gpt2_config_path ${CONFIG} \
--gpt2_model ${INIT_MODEL} \
--learning_rate 5e-5 \
--warmup_steps 0 \
--adam_epsilon 1e-8 \
--module_dir ${MODEL_DIR} \
--module_name ${MODEL} \
--schedule_fname ${SCHEDULED} \
--num_epochs 1 \
--num_iters 2 \
--seed ${SEED} \
--output_dir ${OUT_DIR} \
--no_p2p_prerecv \
--no_prefetch_msgx
# |& tee gpt2_medium_output_BWDvt_打印时间戳.txt
# --nvprof \
# --nvprof_iter "all"
# |& tee ${OUT_DIR}/log.txt
done


# NOTE:
# -. Initial models can be downloaded here (https://1drv.ms/u/s!ApfNYtXZyxcLcKg9KbddiUGFp9E?e=WqcHe2).
# 
# -. In case of hardware randomness, use following flags (at cost of speed):
#       export CUDA_LAUNCH_BLOCKING=1 # single-GPU & Harmony DP only
#       --seed_cudnn
#       --no_all_prefetch_offload
#
# -. Losses need to be moving-averaged.
