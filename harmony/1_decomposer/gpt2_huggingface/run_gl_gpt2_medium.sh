#!/bin/bash
MODEL="gpt2_medium"
OUT_DIR="${PWD}/../../results/${MODEL}"
mkdir -p ${OUT_DIR}

# ----- GPT2 Medium -----
# echo "Clean Python Processes"
# sleep 1s && pkill -9 python3 && pkill -9 python && sleep 1s

# 通过替换模型每一层的前向执行函数，跑一次FWD，在这个过程中创建node（层）和node之间的指向关系（保存在graph实例中），
# 最后将graph字符串化写入txt文件
echo "Creating Graph"
python3 test_gl_gpt2.py \
--train_data_file "../../../data/wikitext-103-tokens/wiki.train.tokens" \
--model_type "gpt2" \
--config_name "../../../model_lib/gpt2_configs/gpt2-medium-config.json" \
--tokenizer_name "gpt2-medium" \
--model_name_or_path "gpt2-medium" \
--cache_dir "../../../pretrained" \
--do_train \
--per_gpu_train_batch_size 1 \
--max_steps 1 \
--logging_steps 1 \
--save_steps -1 \
--no_cuda \
--output_dir ${OUT_DIR} \
--overwrite_output_dir \
--graph_dir ${OUT_DIR} \

# echo
# echo "Generating Code"
# pushd . && cd ..
# python3 code_generator.py \
# --input_dir ${OUT_DIR} \
# --output_dir ${OUT_DIR} \
# --arch ${MODEL} \
# --verbose
# popd

