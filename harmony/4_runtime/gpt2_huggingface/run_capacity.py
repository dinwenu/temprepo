#!/usr/bin/env python3
"""Run the SME/Harmony maximum-model-size matrix from Section 4.2."""
import argparse
import csv
import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

from capacity_memory import inspect_capacity
from prepare_capacity_schedules import prepare
from run_experiment import HERE, MODELS, ROOT


def classify(returncode, text, timeout=False):
    if timeout:
        return 'timeout'
    # Some multiprocessing parents exit zero after a rank fails; inspect OOM evidence first.
    if re.search(r'CUDA out of memory|CUDA error: out of memory|torch\.cuda\.OutOfMemoryError', text, re.I):
        return 'gpu_oom'
    if re.search(r'MemoryError|DefaultCPUAllocator.*(?:allocate|memory)|std::bad_alloc|Cannot allocate memory', text, re.I):
        return 'cpu_oom'
    if returncode == 0:
        # A launcher that exits without the requested number of updates is not a completed run.
        return 'completed' if re.search(r'\[Global\].*Num Updates:\s*[1-9]\d*', text) else 'incomplete'
    return 'failed'


def stop_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--memory-gb', type=int, choices=[128, 256], default=128)
    parser.add_argument('--memory-limit-source', choices=['numa', 'cgroup'], default='numa')
    parser.add_argument('--cpu-nodes')
    parser.add_argument('--memory-nodes')
    parser.add_argument('--methods', '--method', nargs='+', choices=['sme', 'harmony'], default=['sme', 'harmony'])
    parser.add_argument('--models', '--model', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--modes', nargs='+', choices=['PP', 'DP'], default=['PP', 'DP'])
    parser.add_argument('--num-iters', type=int, default=4)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/wikitext-103-tokens/wiki.train.tokens')
    parser.add_argument('--module-dir', type=Path, default=ROOT / 'harmony/results')
    parser.add_argument('--nvme-path', type=Path, default=Path(os.environ.get('SME_NVME_PATH', '/tmp/sme-nvme')))
    parser.add_argument('--output-dir', type=Path, help='New directory for this capacity sweep.')
    parser.add_argument('--timeout-seconds', type=int, default=3600, help='Per-workload time limit; not a paper runtime estimate.')
    parser.add_argument('--check', action='store_true', help='Check all schedules/inputs and print commands without running training.')
    args = parser.parse_args()
    if args.num_iters < 2 or args.timeout_seconds < 1:
        parser.error('At least two iterations and a positive timeout are required')
    args.cpu_nodes = (args.cpu_nodes or '0') if args.memory_gb == 128 else args.cpu_nodes
    args.memory_nodes = (args.memory_nodes or '0') if args.memory_gb == 128 else args.memory_nodes
    try:
        prepare(args.module_dir.resolve(), check=True)
        memory = inspect_capacity(args.memory_gb, args.memory_limit_source, args.cpu_nodes, args.memory_nodes)
    except (ValueError, OSError, StopIteration) as error:
        parser.error(str(error))
    print('Memory constraints: ' + json.dumps(memory), flush=True)
    if memory['errors'] and not args.check:
        parser.error('Capacity memory check failed: ' + '; '.join(memory['errors']))
    if memory['errors']:
        print('CHECK ONLY: the current host does not satisfy the requested memory budget; actual training would refuse it.', flush=True)
    if not args.check and not args.data.is_file():
        parser.error('Dataset is not installed; follow the README download/conversion instructions')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output_dir or HERE / 'logs/section4_2' / f'{args.memory_gb}gb' / stamp).resolve()
    if not args.check:
        output.mkdir(parents=True, exist_ok=False)
        (output / 'constraints.json').write_text(json.dumps(memory, indent=2) + '\n')
    rows = []
    for method in args.methods:
        for model in args.models:
            for mode in args.modes:
                case = f'{method}_{model}_{mode}'
                destination = output / case
                command = [sys.executable, str(HERE / 'run_experiment.py'), '--model', model, '--method', method,
                           '--schedule', f'D4_v{mode}_N4_Ufwd1_Ubwd1_P1', '--capacity',
                           '--memory-gb', str(args.memory_gb),
                           '--num-iters', str(args.num_iters), '--data', str(args.data.resolve()),
                           '--module-dir', str(args.module_dir.resolve()), '--nvme-path', str(args.nvme_path.resolve()),
                           '--output-dir', str(destination)]
                if args.memory_gb == 128:
                    command.extend(['--memory-limit-source', args.memory_limit_source, '--cpu-nodes', args.cpu_nodes, '--memory-nodes', args.memory_nodes])
                if args.check:
                    print('CHECK ' + case, flush=True)
                    subprocess.run(command + ['--check'], check=True)
                    continue
                logfile = output / (case + '.log')
                (output / (case + '.command.json')).write_text(json.dumps(command, indent=2) + '\n')
                print('RUN ' + case + ' -> ' + str(logfile), flush=True)
                timed_out = False
                with logfile.open('w') as stream:
                    process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        process.wait(timeout=args.timeout_seconds)
                    except subprocess.TimeoutExpired:
                        timed_out = True
                        stop_group(process)
                    except KeyboardInterrupt:
                        stop_group(process)
                        raise
                    finally:
                        # Reap any training ranks left behind after a parent error/timeout.
                        stop_group(process)
                text = logfile.read_text(errors='replace')
                status = classify(process.returncode, text, timed_out)
                updates = [int(n) for n in re.findall(r'\[Global\].*Num Updates:\s*(\d+)', text)]
                if status == 'completed' and args.num_iters not in updates:
                    status = 'incomplete'
                rows.append({'memory_gib': args.memory_gb, 'method': method, 'model': model, 'mode': mode,
                             'status': status, 'returncode': process.returncode, 'log': str(logfile)})
                with (output / 'summary.csv').open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(status.upper() + ' ' + case, flush=True)
    if args.check:
        print('All selected packaged workloads checked; no training or memory-limit changes were performed.')
        return 0
    print('Results: ' + str(output / 'summary.csv'))
    # Known OOM cases are expected capacity boundaries; unknown errors need investigation.
    return int(any(row['status'] in ('failed', 'timeout', 'incomplete') for row in rows))


if __name__ == '__main__':
    sys.exit(main())
