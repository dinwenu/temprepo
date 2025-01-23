#!/bin/bash
MODEL_DIR="../results"
for MODEL in "gpt2_2-5b" # ""gpt2_2-5b" # "gpt2_5b" "gpt2_7-5b" "gpt2_10b" "gpt2_12-5b" "gpt2_15b"
do
  
  # -------------- Manual ---------------------------
  for D in 4
  do
    for MODE in "vPP" "vDP"
    do
      for N in 4
      do
      echo "Manual"
      python3 scheduler.py \
      --manual \
      --manual_ufwd 1 \
      --manual_ubwd 1 \
      --manual_packsize 1 \
      --module_dir ${MODEL_DIR} \
      --module_name ${MODEL} \
      --minibatchsize ${D} \
      --mode ${MODE} \
      --num_gpus ${N} \
      --suffix "_gpt2_2-5b_3090"
      #   --simulation
      done
    done
  done

done
