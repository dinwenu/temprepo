#!/bin/bash
echo "Within nvidia's pytorch container, install following requirements"
set -x
# For MKL
conda install mkl mkl-include -y
conda install -c mingfeima mkldnn -y 
# For dot graph
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple graphviz  # torchviz
# For GPT2 
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple tokenizers dataclasses