"""Exercise real worker timing/output paths with CUDA and communication mocked."""
import contextlib
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
for path in ['harmony/4_runtime', 'harmony/3_scheduler', 'harmony/2_profiler', 'util_lib', 'model_lib']:
    sys.path.insert(0, str(ROOT / path))
import worker
import worker_5_layer


class ThroughputTests(unittest.TestCase):
    def test_each_iteration_prints_global_minibatch_throughput(self):
        for module in [worker, worker_5_layer]:
            with self.subTest(worker=module.__name__):
                instance = module.Worker.__new__(module.Worker)
                instance.args = SimpleNamespace(no_update=False, num_iters=2, num_epochs=1,
                                                empty_cache=False, synthetic_data=True)
                instance.rank = 1
                instance.verbose = instance.nvprof = False
                instance.CONFIGS = {'D': 32, 'N': 4, 'mode': 'vPP', 'loss_rank': 0}
                instance.rTASKS = {1: []}
                instance.data_loader = [0, 1]
                instance.data_ubatches = instance.target_ubatches = []
                instance.rand_state_train = Mock()
                instance.update_handler = Mock()
                buffer = io.StringIO()
                with contextlib.ExitStack() as stack:
                    stack.enter_context(contextlib.redirect_stdout(buffer))
                    for name in ['synchronize', 'empty_cache', 'reset_peak_memory_stats']:
                        stack.enter_context(patch.object(module.torch.cuda, name))
                    for name in ['memory_allocated', 'memory_reserved']:
                        stack.enter_context(patch.object(module.torch.cuda, name, return_value=0))
                    stack.enter_context(patch.object(module.dist, 'barrier'))
                    stack.enter_context(patch.object(module.gc, 'collect'))
                    stack.enter_context(patch.object(module, 'CheckGPUMem'))
                    stack.enter_context(patch.object(module, 'pc', side_effect=[0.0, 2.0, 2.0, 6.0]))
                    instance.run_training_loop()
                output = buffer.getvalue()
                self.assertIn('Throughput 16.000 sample/s', output)
                self.assertIn('Throughput 8.000 sample/s', output)
                self.assertEqual(instance.time_iters, [2.0, 4.0])
                self.assertEqual(instance.update_cnt, 2)

    def finish_output(self, module, durations):
        instance = module.Worker.__new__(module.Worker)
        instance.rank = 0
        instance.verbose = False
        instance.CONFIGS = {'D': 32}
        instance.args = SimpleNamespace(save_final_model=False)
        instance.time_iters = durations
        instance.update_cnt = len(durations)
        instance.pcm = Mock()
        instance.pcm.system_cpu_memory.return_value = 'Occupied 64G'
        instance.log_print = lambda message: print(message, flush=True)
        instance.close_log_file = Mock()
        buffer = io.StringIO()
        with patch.object(module.torch.cuda, 'memory_reserved', return_value=0), \
             patch.object(module, 'gather_integer', return_value=[0, 0, 0, 0]), \
             contextlib.redirect_stdout(buffer):
            instance.finish()
        return buffer.getvalue()

    def test_average_uses_second_half_time_not_average_of_rates(self):
        for module in [worker, worker_5_layer]:
            for durations, window in [([100.0, 200.0, 2.0, 4.0], 'Iter[2,4)'),
                                      ([100.0, 2.0, 4.0], 'Iter[1,3)')]:
                with self.subTest(worker=module.__name__, durations=durations):
                    output = self.finish_output(module, durations)
                    self.assertIn(window, output)
                    self.assertIn('Avg Iter Time: 3.000 sec', output)
                    self.assertIn('Avg Throughput: 10.667 sample/s', output)

    def test_empty_iteration_history_preserves_memory_summary(self):
        for module in [worker, worker_5_layer]:
            output = self.finish_output(module, [])
            self.assertIn('Num Updates: 0', output)
            self.assertNotIn('Avg Throughput', output)


if __name__ == '__main__':
    unittest.main()
