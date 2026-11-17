#!/usr/bin/env python3
"""Launch the paper workloads while retaining serialized artifact identifiers."""
import argparse
import datetime
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

MODELS = {
    '1.5b': ('gpt2_xl', 'gpt2-xl-config.json', 1600, 48),
    '2.6b': ('gpt2_2-5b', 'gpt2-2-5-billion.json', 2048, 48),
    '5.9b': ('gpt2_5b', 'gpt2-5-billion.json', 3072, 49),
    '8b': ('gpt2_7-5b', 'gpt2-7-5-billion.json', 3584, 50),
    '10.5b': ('gpt2_10b', 'gpt2-10-billion.json', 4096, 50),
}
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=MODELS)
    parser.add_argument('--method', choices=['sme', 'harmony'], default='sme')
    parser.add_argument('--schedule', default='D16_vPP_N4_Top1')
    parser.add_argument('--capacity', action='store_true', help='Require a verified D4/N4/Ufwd1/Ubwd1/P1 capacity schedule and memory budget.')
    parser.add_argument('--memory-gb', type=int, choices=[128, 256])
    parser.add_argument('--memory-limit-source', choices=['numa', 'cgroup'], default='numa')
    parser.add_argument('--cpu-nodes', help='Comma-separated NUMA CPU nodes for capacity runs.')
    parser.add_argument('--memory-nodes', help='Comma-separated NUMA memory nodes for capacity runs.')
    parser.add_argument('--num-iters', type=int, default=10)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/wikitext-103-tokens/wiki.train.tokens')
    parser.add_argument('--module-dir', type=Path, default=ROOT / 'harmony/results')
    parser.add_argument('--nvme-path', type=Path, default=Path(os.environ.get('SME_NVME_PATH', '/tmp/sme-nvme')))
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path, help='Optional directory containing a matching pytorch_model.bin; otherwise initialize weights from the configuration.')
    parser.add_argument('--check', action='store_true', help='Check packaged inputs and print the command without importing CUDA or running training.')
    args = parser.parse_args()
    if args.num_iters < 2:
        parser.error('--num-iters must be at least 2')
    match = re.fullmatch(r'D(\d+)_v(PP|DP)_N4_(?:Top1|Ufwd1_Ubwd1_P1)', args.schedule)
    if not match:
        parser.error('--schedule must select a four-GPU Top1 or Ufwd1_Ubwd1_P1 schedule')
    model, config_name, width, layers = MODELS[args.model]
    config = ROOT / 'model_lib/gpt2_configs' / config_name
    module_dir = args.module_dir.resolve()
    suffix = '_' + model + '_3090'
    module = module_dir / model
    needed = [config, module / 'code/__init__.py',
              module / 'my_prof' / ('prof_XMETA' + suffix + '.pickle'),
              module / 'my_prof' / ('prof_TMETA' + suffix + '.pickle'),
              module / 'my_sched' / (args.schedule + suffix + '.pickle')]
    missing = [str(p) for p in needed if not p.is_file()]
    if missing:
        parser.error('Missing packaged input(s):\n' + '\n'.join(missing))
    capacity = None
    if args.capacity:
        from schedule_inspect import inspect_schedule
        from capacity_memory import inspect_capacity
        if args.memory_gb is None:
            parser.error('--capacity requires --memory-gb 128 or 256')
        try:
            schedule_report = inspect_schedule(needed[-1], require_microbatch_one=True)
            if (schedule_report['D'], schedule_report['N'], schedule_report['max_pack_size']) != (4, 4, 1) or not args.schedule.endswith('Ufwd1_Ubwd1_P1'):
                raise ValueError('Capacity experiments require D4/N4/Ufwd1/Ubwd1/P1')
            args.cpu_nodes = (args.cpu_nodes or '0') if args.memory_gb == 128 else args.cpu_nodes
            args.memory_nodes = (args.memory_nodes or '0') if args.memory_gb == 128 else args.memory_nodes
            memory_report = inspect_capacity(args.memory_gb, args.memory_limit_source, args.cpu_nodes, args.memory_nodes)
        except (ValueError, OSError) as error:
            parser.error(str(error))
        capacity = {'schedule': schedule_report, 'memory': memory_report}
        print('Capacity schedule: ' + json.dumps(schedule_report), flush=True)
        print('Memory constraints: ' + json.dumps(memory_report), flush=True)
        if memory_report['errors'] and not args.check:
            parser.error('Capacity memory check failed: ' + '; '.join(memory_report['errors']))
        if memory_report['errors']:
            print('Input-check mode only: actual training would refuse the above memory constraints.', flush=True)
        if args.memory_gb == 128 and not args.check and shutil.which('numactl') is None:
            parser.error('numactl is required for the 128 GB capacity runs')
    elif args.memory_gb is not None or args.cpu_nodes or args.memory_nodes:
        parser.error('Memory/NUMA options require --capacity')
    cfg = json.loads(config.read_text())
    if (cfg['n_embd'], cfg['n_layer'], cfg['n_ctx']) != (width, layers, 1024):
        parser.error('Model configuration does not match the paper')
    data = args.data.resolve()
    if args.checkpoint and not (args.checkpoint / 'pytorch_model.bin').is_file():
        parser.error('Checkpoint directory must contain pytorch_model.bin')
    if not args.check and not data.is_file():
        parser.error('Dataset not found: ' + str(data) + '; see README.md')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = (args.output_dir or HERE / 'logs' / args.method / args.model / args.schedule / stamp).resolve()
    command = [sys.executable, 'main_work5.py' if args.method == 'sme' else 'main.py',
               '--gpt2_train_file', str(data), '--gpt2_config_path', str(config), '--gpt2_model', str(args.checkpoint.resolve()) if args.checkpoint else '',
               '--module_dir', str(module_dir), '--module_name', model,
               '--schedule_fname', args.schedule, '--num_iters', str(args.num_iters),
               '--seed_cudnn', '--output_dir', str(output), '--suffix', suffix]
    # Preserve the original NUMA scripts: SME disables model prefetch; Harmony retains it.
    if args.method == 'sme' or not args.capacity:
        command.append('--no_prefetch_model')
    if args.capacity and args.memory_gb == 128:
        command = ['numactl', '--cpunodebind=' + args.cpu_nodes, '--membind=' + args.memory_nodes] + command
    print('Packaged model, configuration, profiles, and schedule: OK', flush=True)
    print('Dataset: ' + str(data) + (' (present)' if data.is_file() else ' (not installed)'), flush=True)
    print('Working directory: ' + str(HERE), flush=True)
    print('Command: ' + shlex.join(command), flush=True)
    if args.check:
        print('Input check only; training and runtime dependencies have not been tested.')
        return 0
    env = os.environ.copy()
    env.setdefault('CUDA_VISIBLE_DEVICES', '0,1,2,3')
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['SME_NVME_PATH'] = str(args.nvme_path.resolve())
    if args.method == 'sme':
        args.nvme_path.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'command.json').open('w') as f:
        json.dump({'command': command, 'cwd': str(HERE), 'SME_NVME_PATH': env['SME_NVME_PATH'],
                   'CUDA_VISIBLE_DEVICES': env['CUDA_VISIBLE_DEVICES'], 'capacity': capacity}, f, indent=2)
    print('Capture console output with tee to retain CPU memory and throughput reports.', flush=True)
    return subprocess.run(command, cwd=HERE, env=env).returncode


if __name__ == '__main__':
    sys.exit(main())
