## Dataset
- WikiText-103: It can be downloaded from [here](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) and unpacked to a directorary `./data/wikitext-103-tokens`.

## Prepare
- The conda environment configuration can be viewed in `environment.yml`.
- Please copy `utils.py` and `partitioned_param_swapper_2.py` to `/home/your-name/miniconda3/envs/your-env-name/lib/python3.9/site-packages/deepspeed/runtime/swap_tensor`

## Experiments
To conduct the experiments in the paper, the scripts are provided in `/harmony/4_runtime/gpt2_huggingface`