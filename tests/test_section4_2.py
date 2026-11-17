"""CPU-only checks for Section 4.2 schedules, memory validation, and sweep results."""
import contextlib
import csv
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / 'harmony/4_runtime/gpt2_huggingface'
sys.path.insert(0, str(ENTRY))
import capacity_memory
import run_capacity
import run_experiment
from prepare_capacity_schedules import layer_count
from schedule_inspect import inspect_schedule, read_schedule


class ScheduleTests(unittest.TestCase):
    def test_all_capacity_schedules(self):
        for size, (model, _, _, _) in run_experiment.MODELS.items():
            module = ROOT / 'harmony/results' / model
            for mode in ('vPP', 'vDP'):
                path = module / 'my_sched' / f'D4_{mode}_N4_Ufwd1_Ubwd1_P1_{model}_3090.pickle'
                with self.subTest(model=size, mode=mode):
                    report = inspect_schedule(path, True)
                    self.assertEqual((report['D'], report['N'], report['u_fwd'], report['u_bwd']), (4, 4, 1, 1))
                    self.assertEqual(report['task_microbatch_sizes'], [1])
                    self.assertEqual(report['max_pack_size'], 1)
                    self.assertEqual(report['R'], layer_count(module))

    def test_original_manual_schedules_preserved(self):
        for model, mode, digest in [
            ('gpt2_2-5b', 'vPP', '7625841272f43ff4b7acdeead1f3cfc21ef3328c33e6530c143ad866c41721d3'),
            ('gpt2_2-5b', 'vDP', 'e87e32ab2a12768d41074a6862767eb0911961dfb8e59eff3afa90281c93d2eb'),
            ('gpt2_5b', 'vPP', 'e4d47f61ef7b2cc103bf388db53fb86dd13d19fac600b7b1f24a60bb62eb2cea'),
            ('gpt2_5b', 'vDP', 'c0d62534be6afbfc163d646af02fae898cb2d8d0dba2c66f709229d59593195e'),
        ]:
            path = ROOT / 'harmony/results' / model / 'my_sched' / f'D4_{mode}_N4_Ufwd1_Ubwd1_P1_{model}_3090.pickle'
            self.assertEqual(inspect_schedule(path, True)['sha256'], digest)

    def test_top1_is_not_a_capacity_schedule(self):
        path = ROOT / 'harmony/results/gpt2_xl/my_sched/D16_vPP_N4_Top1_gpt2_xl_3090.pickle'
        with self.assertRaisesRegex(ValueError, 'microbatch'):
            inspect_schedule(path, True)

    def test_unknown_global_is_rejected_without_execution(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'unsupported.pickle'
            path.write_bytes(b'cbuiltins\nlen\n.')
            with self.assertRaisesRegex(ValueError, 'Unsupported serialized global'):
                read_schedule(path)


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.write('/proc/self/cgroup', '0::/experiment\n')
        self.write('/proc/meminfo', f'MemTotal: {256 * 1024 ** 2} kB\nSwapTotal: 0 kB\n')
        self.write('/sys/fs/cgroup/memory.max', 'max\n')
        self.write('/sys/fs/cgroup/memory.swap.max', 'max\n')
        self.write('/sys/fs/cgroup/experiment/memory.max', 'max\n')
        self.write('/sys/fs/cgroup/experiment/memory.swap.max', '0\n')
        self.set_node(0, 128)
        self.set_node(1, 128)
        patcher = patch.object(capacity_memory, 'Path', side_effect=lambda value: self.root / str(value).lstrip('/'))
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, name, value):
        path = self.root / name.lstrip('/')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)

    def set_node(self, node, size):
        self.write(f'/sys/devices/system/node/node{node}/meminfo', f'Node {node} MemTotal: {size * 1024 ** 2} kB\n')

    def test_matching_physical_node_budgets(self):
        self.assertFalse(capacity_memory.inspect_capacity(128, 'numa', '0', '0')['errors'])
        self.assertFalse(capacity_memory.inspect_capacity(256, 'numa', None, None)['errors'])

    def test_large_node_is_not_a_128g_limit(self):
        self.set_node(0, 1024)
        self.assertTrue(capacity_memory.inspect_capacity(128, 'numa', '0', '0')['errors'])

    def test_collective_cgroup_limit_on_large_node(self):
        self.set_node(0, 1024)
        self.write('/sys/fs/cgroup/experiment/memory.max', str(128 * 1024 ** 3))
        self.assertFalse(capacity_memory.inspect_capacity(128, 'cgroup', '0', '0')['errors'])
        self.write('/sys/fs/cgroup/experiment/memory.max', str(64 * 1024 ** 3))
        self.assertTrue(capacity_memory.inspect_capacity(128, 'cgroup', '0', '0')['errors'])

    def test_full_memory_needs_no_numa_nodes(self):
        with patch.object(capacity_memory, 'node_ids', side_effect=AssertionError('NUMA lookup should not run')):
            report = capacity_memory.inspect_capacity(256, 'numa', None, None)
        self.assertEqual(report['limit_source'], 'system')
        self.assertFalse(report['errors'])

    def test_ancestor_limit_cannot_be_ignored(self):
        self.write('/sys/fs/cgroup/memory.max', str(64 * 1024 ** 3))
        self.write('/sys/fs/cgroup/experiment/memory.max', str(128 * 1024 ** 3))
        self.assertTrue(capacity_memory.inspect_capacity(128, 'cgroup', '0', '0')['errors'])

    def test_swap_cannot_extend_memory_capacity(self):
        self.write('/proc/meminfo', 'SwapTotal: 1048576 kB\n')
        self.write('/sys/fs/cgroup/experiment/memory.swap.max', 'max\n')
        self.assertTrue(capacity_memory.inspect_capacity(128, 'numa', '0', '0')['errors'])


class SweepTests(unittest.TestCase):
    def test_failure_classification(self):
        self.assertEqual(run_capacity.classify(1, 'CUDA out of memory'), 'gpu_oom')
        self.assertEqual(run_capacity.classify(1, 'MemoryError'), 'cpu_oom')
        self.assertEqual(run_capacity.classify(0, 'CUDA out of memory'), 'gpu_oom')
        self.assertEqual(run_capacity.classify(0, 'MemoryError'), 'cpu_oom')
        self.assertEqual(run_capacity.classify(-9, ''), 'failed')
        self.assertEqual(run_capacity.classify(1, 'No module named boto3'), 'failed')
        self.assertEqual(run_capacity.classify(0, ''), 'incomplete')
        self.assertEqual(run_capacity.classify(0, '[Global] Num Updates: 4'), 'completed')
        self.assertEqual(run_capacity.classify(-15, '', timeout=True), 'timeout')

    def test_matrix_logs_success_and_continues_after_oom(self):
        class FakeProcess:
            count = 0
            def __init__(self, command, stdout, **kwargs):
                self.pid = 123
                self.returncode = 0 if FakeProcess.count == 0 else 1
                FakeProcess.count += 1
                stdout.write('[Global] Num Updates: 4\n' if not self.returncode else 'CUDA out of memory\n')
            def wait(self, timeout=None):
                return self.returncode
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'sweep'
            data = Path(folder) / 'wiki.train.tokens'
            data.write_text('fixture')
            args = ['run_capacity.py', '--methods', 'sme', 'harmony', '--models', '1.5b', '--modes', 'PP',
                    '--output-dir', str(out), '--data', str(data)]
            with patch.object(sys, 'argv', args), patch.object(run_capacity, 'prepare'), \
                 patch.object(run_capacity, 'inspect_capacity', return_value={'errors': []}), \
                 patch.object(run_capacity.subprocess, 'Popen', FakeProcess), \
                 patch.object(run_capacity, 'stop_group'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(run_capacity.main(), 0)
            with (out / 'summary.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row['status'] for row in rows], ['completed', 'gpu_oom'])
            self.assertTrue((out / 'constraints.json').is_file())

    def test_256gb_command_runs_python_directly(self):
        for method in ['sme', 'harmony']:
            args = ['run_experiment.py', '--method', method, '--model', '1.5b', '--capacity', '--memory-gb', '256',
                    '--schedule', 'D4_vPP_N4_Ufwd1_Ubwd1_P1', '--check']
            buffer = io.StringIO()
            with patch.object(sys, 'argv', args), contextlib.redirect_stdout(buffer):
                self.assertEqual(run_experiment.main(), 0)
            command = next(line for line in buffer.getvalue().splitlines() if line.startswith('Command:'))
            self.assertTrue(command.startswith('Command: ' + sys.executable + ' '))
            self.assertNotIn('numactl', command)
            self.assertNotIn('--cpunodebind', command)
            self.assertNotIn('--membind', command)

    def test_capacity_command_preserves_prefetch_settings(self):
        for method in ['sme', 'harmony']:
            args = ['run_experiment.py', '--method', method, '--model', '1.5b', '--capacity', '--memory-gb', '128',
                    '--schedule', 'D4_vPP_N4_Ufwd1_Ubwd1_P1', '--check']
            buffer = io.StringIO()
            with patch.object(sys, 'argv', args), contextlib.redirect_stdout(buffer):
                self.assertEqual(run_experiment.main(), 0)
            command = next(line for line in buffer.getvalue().splitlines() if line.startswith('Command:'))
            self.assertIn('numactl --cpunodebind=0 --membind=0', command)
            self.assertEqual('--no_prefetch_model' in command, method == 'sme')


if __name__ == '__main__':
    unittest.main()
