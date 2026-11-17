#!/usr/bin/env python3
"""Prepare and inspect Section 4.2 schedules: D=4, N=4, Ufwd=Ubwd=1, P=1."""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from run_experiment import MODELS, ROOT
from schedule_inspect import inspect_schedule


def layer_count(module):
    tree = ast.parse((module / 'code/__init__.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'model')
    returned = next(n.value for n in ast.walk(function) if isinstance(n, ast.Return))
    if not isinstance(returned, ast.List):
        raise ValueError('Expected an explicit model layer list')
    return len(returned.elts)


def prepare(module_dir, check=False):
    sys.path.insert(0, str(ROOT / 'harmony/3_scheduler'))
    from task_graph_composor import compose
    from task_data_struct import serialize_scheduled
    records = []
    for size, (model, _, _, _) in MODELS.items():
        module = module_dir / model
        count = layer_count(module)
        for mode in ('vPP', 'vDP'):
            name = f'D4_{mode}_N4_Ufwd1_Ubwd1_P1_{model}_3090'
            path = module / 'my_sched' / (name + '.pickle')
            if not path.is_file():
                if check:
                    raise ValueError('Missing capacity schedule; run this script without --check: ' + str(path))
                args = SimpleNamespace(num_layers=count, mode=mode, num_gpus=4, minibatchsize=4,
                                       reverse_bwd=False, no_offload_optim=False, last_fwd_msg=False)
                pack_bwd = [[index] for index in range(count)]
                pack_fwd = pack_bwd[:-1]
                with contextlib.redirect_stdout(io.StringIO()):
                    configs, _, rtasks = compose(args, 1, pack_fwd, 1, pack_bwd)
                    serialize_scheduled(rtasks, configs, str(module), name, base_dir='my_sched')
            report = inspect_schedule(path, require_microbatch_one=True)
            if (report['N'], report['D'], report['R'], report['mode'], report['max_pack_size']) != (4, 4, count, mode, 1):
                raise ValueError('Capacity schedule does not match the generated model: ' + str(path))
            records.append(dict(report, paper_size=size))
            print(f"OK {size} {mode}: D=4 N=4 Ufwd=1 Ubwd=1 task microbatches={report['task_microbatch_sizes']}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--module-dir', type=Path, default=ROOT / 'harmony/results')
    parser.add_argument('--check', action='store_true', help='Inspect existing schedules without creating files.')
    parser.add_argument('--report', type=Path, help='Optional JSON audit report.')
    args = parser.parse_args()
    try:
        records = prepare(args.module_dir.resolve(), args.check)
    except (ValueError, OSError, StopIteration) as error:
        parser.error(str(error))
    if args.report:
        args.report.write_text(json.dumps(records, indent=2) + '\n')


if __name__ == '__main__':
    main()
