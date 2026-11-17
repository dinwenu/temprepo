# SME

Artifact code for the SSD-based memory-efficient parallel training system in the CGO 2027 paper. SME extends Harmony with task-level parameter, gradient management and layer-granularity NVMe parameter swapping. The paper workloads use FP32, sequence length 1024, four GPUs, and activation recomputation.

## Contents

- `harmony/4_runtime/worker_5_layer.py`: SME runtime worker.
- `harmony/4_runtime/partitioned_param_swapper_2.py`: repository-local NVMe swapper, using DeepSpeed's asynchronous I/O extension.
- `harmony/4_runtime/gpt2_huggingface/`: experiment launchers and GPT-2 training entry points; `main_work5.py` runs SME and `main.py` runs Harmony.
- `harmony/results/<model>/code`, `my_prof`, and `my_sched`: generated model code, profiles, and four-GPU schedules.
- `model_lib/gpt2_configs/`: model configurations.
- `environment.yml`: exported software environment.

## Checklist

- **Algorithm:** Task-level parameter and gradient memory management with layer-granularity NVMe parameter swapping.
- **Program:** SME and Harmony GPT-2 training workloads; source code, generated model code, profiles, schedules, and experiment scripts are included.
- **Models:** GPT-2 variants with 1.5B, 2.6B, 5.9B, 8.0B, and 10.5B parameters, sequence length 1024, and FP32 precision. Configurations are included; pretrained weights are not bundled. Default runs initialize weights from the configurations.
- **Data set:** WikiText-103, `wikitext-103-v1`, downloaded separately from the pinned Hugging Face revision linked below. Approximately 313 MB of Parquet downloads and 542 MB of reconstructed text; conversion code and SHA256 verification are provided. The GPT-2 tokenizer is downloaded separately on first use.
- **Compilation:** A C++ compiler, a CUDA toolkit/compiler compatible with PyTorch, Ninja, and libaio headers/libraries are required for DeepSpeed's I/O extension.
- **Run-time environment:** Linux x86-64; Python 3.9.19, PyTorch 1.12.1, CUDA 11.3, DeepSpeed 0.11.1, and dependencies in `environment.yml`.
- **Hardware:** Reference server: four RTX 3090 GPUs with 24 GB each, two 12-core Intel Xeon 4214R CPUs, 256 GB DRAM, a 2 TB Samsung PM981 NVMe SSD, and PCIe 3.0 x16 GPU connections.
- **Run-time state:** Throughput averages exclude the first half of completed training iterations as warm-up.
- **Execution:** Run workloads sequentially on four GPUs with activation recomputation. The 256 GB experiment runs directly on the full-memory reference server; the 128 GB experiment uses NUMA binding or a collectively limited, no-swap cgroup.
- **Metrics:** CPU memory consumption, GPU memory usage, iteration time, throughput in samples/s, training loss, and maximum trainable model size.
- **Output:** Per-iteration loss/throughput and final average-throughput summaries; loss arrays and `command.json` in experiment output directories. Capacity sweeps additionally save case logs, `constraints.json`, and `summary.csv` with completion/failure status.
- **Experiments:** Setup, basic checks, commands, and expected results are documented below.
- **Result variability:** Absolute timings and throughput depend on CPU/GPU/PCIe/SSD performance and system load. No empirically validated numerical tolerance has been established; compare memory-efficiency trends and capacity boundaries under matching conditions.
- **Disk space:** We recommend at least 150 GB of free SSD space.

## Requirements and installation

The paper's server has four RTX 3090 GPUs (24 GB each), two 12-core Intel Xeon 4214R CPUs at 2.40 GHz, 256 GB DRAM, and a 2 TB Samsung PM981 NVMe SSD, with PCIe 3.0 x16 connections. The paper reports PyTorch 1.12.1, CUDA 11.3, and NCCL 2.10. Use Linux, a compatible NVIDIA driver, a CUDA toolkit/compiler for extension builds, a C++ compiler, Ninja, and libaio headers/libraries.

From the repository root:

```bash
conda env create -n sme -f environment.yml
conda activate sme
# Install libaio development headers/libraries, e.g. on Debian/Ubuntu:
# sudo apt-get install libaio-dev
export LIBRARY_PATH="$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CPATH="$CONDA_PREFIX/include:${CPATH:-}"
ulimit -n 51200
python3 -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.nccl.version()); assert torch.cuda.device_count() >= 4'
python3 -c 'from deepspeed.ops.op_builder import AsyncIOBuilder; AsyncIOBuilder().load()'
```

The open-file limit must be permitted by the shell's hard limit. DeepSpeed's I/O extension is built on first use. The custom swapper is imported from `harmony/4_runtime`; its synchronous read/write helpers are defined locally, while standard DeepSpeed utilities remain unmodified. The obsolete root-level copy helpers have been removed.

Set a writable directory on the NVMe SSD, dedicated to these experiments:

```bash
export SME_NVME_PATH=/path/on/nvme/sme
mkdir -p "$SME_NVME_PATH"
```

Explicitly set the SSD path for measurements. Run experiments sequentially: the swapper initializes and reuses per-rank files under this directory. Reserve space for at least one FP32 parameter image (about 40 GB for the largest workload), plus the dataset, caches, and outputs, with additional headroom.

## Dataset and model initialization

### Download WikiText-103

Use the **`wikitext-103-v1`** subset in the [Salesforce WikiText repository on Hugging Face](https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-v1), pinned to revision `b08601e04326c79dfdd32d625aee71d232d685c3`. Select the tokenized `wikitext-103-v1` subset; `wikitext-103-raw-v1` contains different text.

Hugging Face provides four Parquet files: two training shards, one validation shard, and one test shard. Their `text` column preserves the original nonblank lines, including their trailing newlines, but represents each whitespace-only line as an empty string. Restore each empty string to **one space followed by one newline**, `" \n"`. Concatenate rows in order, processing training shard `00000` before `00001`, and write UTF-8 bytes. Do not use `strip()`, drop empty rows, or insert an additional newline after every row; these operations change the input to the GPT-2 tokenizer.

The procedure below has been verified to reproduce all three original experiment files byte-for-byte, including their SHA256 checksums. Run the commands **from the repository root**.

1. Install the Parquet reader in the activated environment, and download the pinned files (approximately 313 MB in total). Allow approximately 1 GB of free disk space for the downloads and reconstructed text.

```bash
python3 -m pip install "pyarrow>=10,<20"
mkdir -p data/hf-wikitext-103-v1
(
  set -e
  HF_BASE="https://huggingface.co/datasets/Salesforce/wikitext/resolve/b08601e04326c79dfdd32d625aee71d232d685c3/wikitext-103-v1"
  for part in train-00000-of-00002.parquet train-00001-of-00002.parquet validation-00000-of-00001.parquet test-00000-of-00001.parquet; do
    curl --fail --location --retry 3 "${HF_BASE}/${part}" \
      --output "data/hf-wikitext-103-v1/${part}"
  done
)
```

2. Restore the original text files. The following code streams Parquet batches, restores blank lines, verifies each output, and only then moves it into its final location. The Hugging Face `validation` split becomes `wiki.valid.tokens`.

```bash
python3 - <<'PYCODE'
import hashlib
from pathlib import Path
import pyarrow.parquet as pq

source = Path("data/hf-wikitext-103-v1")
output = Path("data/wikitext-103-tokens")
output.mkdir(parents=True, exist_ok=True)
splits = {
    "train": (["train-00000-of-00002.parquet", "train-00001-of-00002.parquet"],
              539209157, "2082f99929dee10a3b96da7d6ef7ca4dbb4d7df9d51db264f14173a36989f826"),
    "valid": (["validation-00000-of-00001.parquet"],
              1142309, "9aee9f52d6e77de6c7751fc5e7c44287db3ec0eaf15477098acc6b93c1c4250b"),
    "test": (["test-00000-of-00001.parquet"],
             1281077, "56db52eb157b1ab5c89646eef70510c1eca6e12a3bfdb99f0d6b05b4eacb75b9"),
}
for split, (shards, expected_size, expected_sha256) in splits.items():
    destination = output / f"wiki.{split}.tokens"
    temporary = destination.with_suffix(".tokens.tmp")
    digest = hashlib.sha256()
    size = 0
    with temporary.open("wb") as stream:
        for shard in shards:
            parquet = pq.ParquetFile(source / shard)
            for batch in parquet.iter_batches(batch_size=8192, columns=["text"]):
                rows = batch.column(0).to_pylist()
                data = "".join(text if text != "" else " \n" for text in rows).encode("utf-8")
                stream.write(data)
                digest.update(data)
                size += len(data)
    if size != expected_size or digest.hexdigest() != expected_sha256:
        temporary.unlink()
        raise RuntimeError(f"Dataset verification failed: {destination}")
    temporary.replace(destination)
    print(f"OK {destination}: {size} bytes, SHA256 {digest.hexdigest()}")
PYCODE
```

Expected output is one `OK` line per file, with these sizes:

| File | Size (bytes) |
| --- | ---: |
| `wiki.train.tokens` | 539,209,157 |
| `wiki.valid.tokens` | 1,142,309 |
| `wiki.test.tokens` | 1,281,077 |

After conversion, the default training path `data/wikitext-103-tokens/wiki.train.tokens` is ready to use. The intermediate `data/hf-wikitext-103-v1/` directory can be deleted after successful verification.

### Tokenizer and model initialization

The GPT-2 tokenizer is downloaded on first use and can be cached beforehand. Dataset and tokenizer downloads are separate from the source package. Feature caches such as `gpt2_cached_lm_1024_wiki.train.tokens` are generated locally from the restored text files.

The included commands pass an empty `--gpt2_model`, so weights are initialized from the configuration. They exercise the training and memory/throughput workloads; they do not download pretrained weights. For checkpoint fine-tuning, supply `--checkpoint /path/to/matching-checkpoint`, whose directory must contain `pytorch_model.bin` matching the selected architecture. In particular, the enlarged variants cannot use an unmodified GPT-2 XL checkpoint. Pretrained weights are not bundled.

## Experiment scripts

All paths in this table are relative to `harmony/4_runtime/gpt2_huggingface/`. Scripts can be launched from any working directory.

| Paper size | SME script | Harmony script | Internal model identifier | Configuration filename |
| --- | --- | --- | --- | --- |
| 1.5B | `run_gpt2_work5_xl.sh` | `run_gpt2_xl.sh` | `gpt2_xl` | `gpt2-xl-config.json` |
| 2.6B | `run_gpt2_work5_2-6b.sh` | `run_gpt2_2-6b.sh` | `gpt2_2-5b` | `gpt2-2-5-billion.json` |
| 5.9B | `run_gpt2_work5_5-9b.sh` | `run_gpt2_5-9b.sh` | `gpt2_5b` | `gpt2-5-billion.json` |
| 8.0B | `run_gpt2_work5_8b.sh` | `run_gpt2_8b.sh` | `gpt2_7-5b` | `gpt2-7-5-billion.json` |
| 10.5B | `run_gpt2_work5_10-5b.sh` | `run_gpt2_10-5b.sh` | `gpt2_10b` | `gpt2-10-billion.json` |

Internal model/configuration names and schedule suffixes intentionally keep their historical spelling; they are identifiers, not parameter-count claims. The architectures use `(hidden dimension, Transformer layers)` of `(1600,48)`, `(2048,48)`, `(3072,49)`, `(3584,50)`, and `(4096,50)`, respectively. Historical workloads outside this table are not part of the paper's main reproduction path.

## Basic checks and short training run

From the repository root, check that a workload's model code, configuration, profiles, and schedule are present:

```bash
bash harmony/4_runtime/gpt2_huggingface/run_gpt2_work5_2-6b.sh --check
```

Expected output includes `Packaged model, configuration, profiles, and schedule: OK` and the training command. This check needs only Python's standard library; it reports dataset presence separately and does not validate GPUs, libaio, or training.

After installing dependencies and the dataset, exercise two iterations of the smallest workload:

```bash
set -o pipefail
bash harmony/4_runtime/gpt2_huggingface/run_gpt2_work5_xl.sh \
  --schedule D16_vPP_N4_Top1 --num-iters 2 2>&1 | tee sme-smoke.log
```

A successful run initializes four ranks, completes two iterations, produces finite losses, prints per-iteration durations and a final GPU/CPU memory summary, and exits successfully. This is a functional check, not a performance result. Its duration depends on tokenizer/data caching, extension compilation, and storage performance; no measured runtime estimate is supplied yet.

## Memory and throughput experiments

Each main launcher defaults to `D16_vPP_N4_Top1` and 10 iterations. Select PP or DP and the minibatch size through the schedule name; these settings are loaded from the serialized schedule, not changed by a separate batch-size flag. The standard matrix contains `D16`, `D32`, `D64`, `D128`, and `D256` for `vPP` and `vDP` on four GPUs. Check availability before running a cell:

```bash
bash harmony/4_runtime/gpt2_huggingface/run_gpt2_work5_5-9b.sh \
  --schedule D32_vDP_N4_Top1 --check
bash harmony/4_runtime/gpt2_huggingface/run_gpt2_work5_5-9b.sh \
  --schedule D32_vDP_N4_Top1 --num-iters 10 2>&1 | tee sme-5.9b-d32-dp.log
bash harmony/4_runtime/gpt2_huggingface/run_gpt2_2-6b.sh \
  --schedule D16_vPP_N4_Top1 --num-iters 10 2>&1 | tee harmony-2.6b-d16-pp.log
```

Use the table to run other model sizes. `--data`, `--module-dir`, `--nvme-path`, `--output-dir`, and `--checkpoint` customize inputs and storage; `--help` lists all options. `CUDA_VISIBLE_DEVICES` defaults to `0,1,2,3` and can be overridden with another four-GPU selection. The launcher sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` and enables cuDNN seeding; deterministic backend settings are also applied in the migrated runtimes/workers.

Outputs are stored separately in `harmony/4_runtime/gpt2_huggingface/logs/<method>/<paper-size>/<schedule>/<UTC timestamp>/`. Each run records `command.json`. Save console output with `tee` to retain initialization memory reports, iteration timings, and the final memory/update summary. Both workers print per-iteration throughput as global minibatch size divided by iteration duration, in samples/s. The final rank-0 summary also prints `Avg Iter Time` and `Avg Throughput`, discarding the first half of completed iterations as warm-up.

For the CPU-memory and throughput figures, compare SME and Harmony with identical models, schedules, precision, data, and machine state. CPU `occupied` reports are **system-wide `total - available` snapshots**, not isolated process-tree peaks. The paper reports OOM for some large-model/batch combinations, including SME 10.5B at minibatches 128 and 256 with 256 GB DRAM.

## Maximum trainable model size

The capacity experiments use **forward and backward microbatch size 1** on four GPUs.

Dedicated schedules now cover **all five model sizes and both PP/DP modes**. They use global minibatch `D=4`, four GPUs (`N4`), microbatch sizes `Ufwd1_Ubwd1`, and one layer per pack (`P1`). With DP, each GPU processes one sample per step; with PP, the global minibatch contains four single-sample microbatches. These schedules are separate from the `Top1` schedules used for other experiments.

From the repository root, inspect all ten schedules:

```bash
python3 harmony/4_runtime/gpt2_huggingface/prepare_capacity_schedules.py --check
# Optionally save a detailed audit report:
python3 harmony/4_runtime/gpt2_huggingface/prepare_capacity_schedules.py \
  --check --report capacity-schedules.json
```

Each line should report `D=4 N=4 Ufwd=1 Ubwd=1 task microbatches=[1]`. Run the same preparation script without `--check` to recreate missing schedules from the generated model code using Harmony's task-graph composer; existing schedules are checked and retained.

### Check NUMA memory capacity

The 256 GB experiment runs directly on the reference server with its full 256 GB of DRAM; it uses no `numactl` command or NUMA-binding arguments. For the 128 GB experiment, install `numactl` and inspect the review machine:

```bash
numactl --hardware
```

**`numactl --membind=0` binds allocations to node 0; it does not independently impose a 128 GB limit.** The 128 GB experiment uses `--cpunodebind=0 --membind=0` on the reference server, where node 0 has approximately 128 GiB. Match this node ID and capacity to the actual machine. For these scripts, `--memory-gb` uses GiB (2^30 bytes); the 128 GB run records node capacity and the 256 GB run records total system memory.

The 128 GB run checks bound-node capacity, inherited memory limits, and swap. The 256 GB run checks the full server memory and any smaller inherited memory limit, without imposing a NUMA policy. The scripts only inspect constraints; they do not change system memory limits or swap settings. For the 128 GB run, NUMA CPU/memory bindings apply to the training process and its spawned ranks, including allocations during initialization.

### Run the 128 GB and 256 GB sweeps

Set `SME_NVME_PATH` to a dedicated directory on the SSD and install the dataset as described above. First check packaged inputs without starting training:

```bash
bash harmony/4_runtime/gpt2_huggingface/run_section4_2_128gb.sh --check
bash harmony/4_runtime/gpt2_huggingface/run_section4_2_256gb.sh --check
```

`--check` verifies all schedules/inputs and prints planned commands and detected memory constraints. It reports memory-budget mismatches but does not start training.

Run the actual experiments sequentially:

```bash
bash harmony/4_runtime/gpt2_huggingface/run_section4_2_128gb.sh
bash harmony/4_runtime/gpt2_huggingface/run_section4_2_256gb.sh
```

Each sweep runs SME and Harmony, five sizes (1.5B, 2.6B, 5.9B, 8.0B, and 10.5B), and PP/DP, for 20 cases per memory budget. Each case defaults to four training iterations. All cases use FP32 and activation recomputation. Cases that run out of memory are part of the capacity comparison.

Select a smaller subset; node overrides apply only to the 128 GB run:

```bash
bash harmony/4_runtime/gpt2_huggingface/run_section4_2_128gb.sh \
  --methods sme --models 5.9b --modes PP DP --num-iters 4 \
  --cpu-nodes 0 --memory-nodes 0
bash harmony/4_runtime/gpt2_huggingface/run_section4_2_256gb.sh \
  --methods harmony --models 2.6b --modes PP DP
```

The model-specific `run_gpt2_work5_<size>_numactl.sh` (SME) and `run_gpt2_<size>_numactl.sh` (Harmony) wrappers are available for `xl`, `2-6b`, `5-9b`, `8b`, and `10-5b`. They default to the 128 GB budget and run both modes. For example:

```bash
bash harmony/4_runtime/gpt2_huggingface/run_gpt2_work5_5-9b_numactl.sh \
  --modes PP
```

`--data`, `--module-dir`, `--nvme-path`, `--output-dir`, and `--timeout-seconds` customize a sweep. `--output-dir` must name a new directory.

### Machines with different NUMA-node sizes

For the 128 GB experiment on a machine with different node sizes, an administrator may provide a **cgroup v2** with `memory.max=128G` and `memory.swap.max=0` for the entire experiment process tree. Then select `--memory-limit-source cgroup` and bind to nodes exposing enough physical memory. For example, inside an already configured 128 GiB no-swap cgroup:

```bash
bash harmony/4_runtime/gpt2_huggingface/run_section4_2_128gb.sh \
  --memory-limit-source cgroup --cpu-nodes 0 --memory-nodes 0
```

## Expected results

 On the paper's reference platform, both SME-PP and SME-DP are expected to support models up to 10.5B with 256 GB CPU memory and 5.9B with 128 GB, compared with 2.6B and 1.5B for Harmony-PP and Harmony-DP, respectively. SME should use less CPU memory for comparable workloads.
