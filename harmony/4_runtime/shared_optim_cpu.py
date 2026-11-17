# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import copy
import gc
import threading
import queue
from collections import OrderedDict as ODict
import torch
from torch.cuda.nvtx import range_push as nvtx_range_push
from torch.cuda.nvtx import range_pop as nvtx_range_pop
from task_data_struct import Medium, vTask
import threadsafe_data_struct
import time
if os.environ.get('CUDA_LAUNCH_BLOCKING') in ['1','True', True]:
    MEMCPY_NONBLK = False
else:
    MEMCPY_NONBLK = True
def convert_to_pinned(local_model_cpu):
    ''' in-place convert a local model cpu to a pinned model (params and buffers: pinned, local, CPU, no grad) '''
    @torch.no_grad()
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # no grad
                assert param.grad is None, "convert to pinned model requires no grad in input model"
                param.detach_()
                assert not param.requires_grad
                # pin param


                param.data = param.pin_memory() # in-place update and let python do the gc
        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                assert not buf.requires_grad # buffer has no grad
                m._buffers[key] = buf.pin_memory() # in-place update and let python do the gc
                assert not m._buffers[key].requires_grad
    local_model_cpu.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    gc.collect()


class SharedOptimCPU(object):
    """ A wrapper class of shared optimizer (referring to each shared vlayer) on CPU.
        Data structure:
        o	a multiprocess-shared vlayer and optimizer
        o	a process-local pinned .grad as input buffer
        o	a process-local pinned model (param and buffs) as output buffer
        TODO: decouple pinned_model from here
    """

    @torch.no_grad()
    def __init__(self, shared_model, optimizer, id=-1):
        """ Call this in parent process and before fork/spawn subprocess """
        self.id = id # vlayer_id
        if optimizer is None: # this wrapper is only for pinned model
            self.shared_model = shared_model
            self.shared_optimizer = None
            # print("[SharedOptimizer][id%d] optimizer is None; for pinned model only."%self.id)
        else:
            # confirm model and optimizer are on CPU

            for param in shared_model.parameters():
                assert not param.is_cuda and param.is_shared()
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    # print("..............:", self.id)
                    # print(k)
                    # print(v)
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
            # 1) create zero gradient

            for param in shared_model.parameters():
                param.grad = torch.zeros_like(param.data) # == torch.zeros(input.size(), input.dtype, input.layout, input.device, requires_grad=False)
                # Note: "param.grad = tensor" works, but "param.data.grad = tensor" doesn't work
                # Note: id(param.grad) == id(param.data.grad)
            # 2) force initialization of optimizer states (Bert, GPT2, Adam, SGD)


            optimizer.step()

            # 3) move optimzer.state to shared memory

            # print("[SharedOptimizer] sharing optimizer:")
            for param, state in optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.).
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():

                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)

                        #
                        v.share_memory_(); assert v.is_shared()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))


                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64)
                        state[k].share_memory_(); assert state[k].is_shared()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            # 4) move optimzer.hyperparams to shared memory? No. They are constant (Bert, GPT2, Adam, SGD)
            # 5) clean up gradient

            for param in shared_model.parameters():
                param.grad = None
                assert param.grad is None, "cannot None grad?"

            gc.collect()
            # optimizer.zero_grad()
            self.shared_model = shared_model
            self.shared_optimizer = optimizer
            # print(f"self.shared_optimizer.param_groups:{self.shared_optimizer.param_groups}")
            # print(f"self.shared_optimizer.param_groups[0]:{self.shared_optimizer.param_groups[0]}")
            # exit(0)
            # print("[SharedOptimizer][id%d] sharing optimizer done."%self.id)


    @torch.no_grad()
    def init_in_subproc(self, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this on entering subprocess """
        self.rank, self.world_size = rank, world_size
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf


        if self.shared_optimizer is not None:
            # confirm model and optimizer are shared
            for param in self.shared_model.parameters():
                assert param.data.is_shared()
                assert param.requires_grad
                # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
            # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.).
                for k, v in state.items():
                    assert isinstance(v, torch.Tensor) and v.is_shared()


            for k, v in self.shared_optimizer.param_groups[0].items():
                if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
                    assert (not isinstance(v, torch.Tensor)) # or (isinstance(v, torch.Tensor) and v.is_shared())
                # ref1: https://pytorch.org/docs/1.5.0/_modules/torch/optim/optimizer.html#Optimizer.state_dict
                # ref2: https://pytorch.org/docs/1.5.0/_modules/torch/optim/adam.html#Adam
            # print("[SharedOptimizer] rank{}'s optimizer is shared".format(self.rank))

            # initialize local pinned .grad # Trimed
            # for param in self.shared_model.parameters():
            #     assert param.requires_grad
            #     param.grad = torch.zeros(param.shape, dtype=param.dtype, device="cpu", requires_grad=False).pin_memory()
            #     assert not param.grad.is_shared() and param.grad.is_pinned()
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned .grad"%(self.id, self.rank))


        if self.no_pin_model:
            self.pinned_model = None

        else:
            # initialize local pinned model (params and buffs)
            # with torch.no_grad():

            self.pinned_model = copy.deepcopy(self.shared_model)

            convert_to_pinned(self.pinned_model) # in-place convert a local model cpu to a pinned model
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned model"%(self.id, self.rank))
            # for s_param, p_param in zip(self.shared_model.parameters(), self.pinned_model.parameters()):
            #     print("Values equal:", torch.all(s_param == p_param))
            #     print("Same object:", s_param.data_ptr() == p_param.data_ptr())
            #     print(p_param)


    # shared_model.named_buffers()
    @torch.no_grad()
    def update_buf(self):
        """ In-place copy buffer from local pinned buf to shared model """
        if self.no_pin_grad_buf:
            return


        else:
            assert hasattr(self.shared_model, "pinned_buf")
            named_pin_buf = self.shared_model.pinned_buf
            for name, shared_buf in self.shared_model.named_buffers():
                if shared_buf is not None:
                    shared_buf.data.copy_(named_pin_buf[name].data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local pinned buffer must match shared model"

        # for shared_buf, pinned_buf in zip(self.shared_model.buffers(), self.pinned_model.buffers()):
        #     shared_buf.data.copy_(pinned_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from pinned memory to shared memory # nonblocking useless
        #     assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)

    @torch.no_grad()
    def step(self, zero_grad=False):
        if self.shared_optimizer is not None:
            self.shared_optimizer.step()
            if zero_grad:

                self.shared_optimizer.zero_grad()
        # confirm local .grad is still pinned
        # for param in self.shared_model.parameters():
        #     assert param.grad.is_pinned()
        # print("[SharedOptimizer] rank{} steped shared optimizer".format(self.rank))


    @torch.no_grad()
    def sync_pinned_model(self):
        """ In-place copy from shared model to local pinned model """
        if self.no_pin_model:
            return

        # for name, param in self.shared_model.named_parameters():
        #     lines.append(f"  {name}: shape={param.shape}, {param}, dtype={param.dtype}, is_pinned={param.is_pinned()}")
        # print("\n".join(lines), flush=True)
        #
        for pinned_param, shared_param in zip(self.pinned_model.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(self.pinned_model.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)


class PinnedBufferInformation_2(object):

    def __init__(self, layer_count, transformer_layer, rank):
        self.layer_count = layer_count
        self.transformer_layer = transformer_layer
        self.rank = rank

        self.swap_element_size = torch.tensor([], dtype=torch.float32).element_size()
        self.aligned_bytes = 1024
        self.numel_alignment = self.aligned_bytes // self.swap_element_size


        self.param_idx_in_layer_to_numel = {}
        self.param_idx_to_start_pos = {}
        self.param_idx_to_shape = {}

        self.layer_size, self.buffer_size = self.cal_pinned_buffer_size()

        self.buffers_size = self.buffer_size * self.layer_count

        self.layer_idx_to_buffer_start_pos = {}

        self.buf_start_pos = 0

        for idx in range(self.layer_count):
            self.layer_idx_to_buffer_start_pos[idx] = idx * self.buffer_size


    def cal_pinned_buffer_size(self):
        total_size = 0

        global_param_idx = 0
        for m in self.transformer_layer.modules():

            for id, (key, param) in enumerate(m._parameters.items()):
                param_size = param.numel()
                self.param_idx_in_layer_to_numel[global_param_idx] = param_size
                self.param_idx_to_shape[global_param_idx] = tuple(param.shape)

                self.param_idx_to_start_pos[global_param_idx] = total_size
                total_size += param_size
                global_param_idx += 1

            for key, buf in m._buffers.items():
                buf_size = buf.numel()
                self.param_idx_in_layer_to_numel[global_param_idx] = buf_size
                self.param_idx_to_shape[global_param_idx] = tuple(buf.shape)
                self.param_idx_to_start_pos[global_param_idx] = total_size
                total_size += buf_size
                global_param_idx += 1

        aligned_total_size = self._io_aligned_numel(total_size)
        print(f"rank:{self.rank}, 单层transformer模型原本的参数量{total_size} ,需要的对齐字参数量:{aligned_total_size}, {total_size*4/1024/1024}MB")

        return total_size, aligned_total_size

    def get_layer_size(self):
        return self.layer_size

    def get_buffer_size(self):
        return self.buffer_size

    def get_buffers_size(self):
        return self.buffers_size

    #
    def _io_aligned_numel(self, numel):

        remainder = numel % self.numel_alignment
        return numel if remainder == 0 else (numel + self.numel_alignment - remainder)

    def get_layer_start_pos_in_buffers(self, layer_idx):
        return self.layer_idx_to_buffer_start_pos[layer_idx]

    def get_param_start_pos(self, param_idx):
        return self.param_idx_to_start_pos[param_idx]

    def get_param_size(self, param_idx):
        return self.param_idx_in_layer_to_numel[param_idx]

    def get_param_shape(self, param_idx):
        return self.param_idx_to_shape[param_idx]


class SharedOptimCPU_for_worker5(object):
    """ A wrapper class of shared optimizer (referring to each shared vlayer) on CPU.
        Data structure:
        o	a multiprocess-shared vlayer and optimizer
        o	a process-local pinned .grad as input buffer
        o	a process-local pinned model (param and buffs) as output buffer
        TODO: decouple pinned_model from here
    """

    @torch.no_grad()
    def __init__(self, shared_model, optimizer, id=-1):
        """ Call this in parent process and before fork/spawn subprocess """
        self.id = id # vlayer_id
        if optimizer is None: # this wrapper is only for pinned model
            self.shared_model = shared_model
            self.shared_optimizer = None
            # print("[SharedOptimizer][id%d] optimizer is None; for pinned model only."%self.id)
        else:
            # confirm model and optimizer are on CPU

            for param in shared_model.parameters():
                assert not param.is_cuda and param.is_shared()
            for param, state in optimizer.state.items():
                for k, v in state.items():
                    # print("..............:", self.id)
                    # print(k)
                    # print(v)
                    if isinstance(v, torch.Tensor):
                        assert not v.data.is_cuda
            # 1) create zero gradient

            for param in shared_model.parameters():
                param.grad = torch.zeros_like(param.data) # == torch.zeros(input.size(), input.dtype, input.layout, input.device, requires_grad=False)
                # Note: "param.grad = tensor" works, but "param.data.grad = tensor" doesn't work
                # Note: id(param.grad) == id(param.data.grad)
            # 2) force initialization of optimizer states (Bert, GPT2, Adam, SGD)


            optimizer.step()

            # 3) move optimzer.state to shared memory

            # print("[SharedOptimizer] sharing optimizer:")
            for param, state in optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.).
                # print("\toptimzer.state[{}]".format(param.data.shape))
                for k, v in state.items():

                    if isinstance(v, torch.Tensor):
                        # num_elements = v.numel()
                        # element_size = v.element_size()
                        # memory_size_mb = (num_elements * element_size) / (1024 * 1024)

                        #
                        v.share_memory_(); assert v.is_shared()
                        # print("\t\t{}: {} moved to shared memory".format(k,v.shape))


                    elif isinstance(v, int): # or isinstance(v, float):
                        # option-1: modify optimizer code by v = mp.Value('i', 0) and v.value +=1 # ref: https://github.com/leonardo0lyj/myPT_V0.4_NetTensor/blob/master/MyCommon/MyCommon.py
                        # option-2*: cast it to scalar tensor for sharing and cast it back with .item() during usage
                        # (checked: BertAdam)
                        state[k] = torch.tensor(v, dtype=torch.int64) # if isinstance(v, int) else torch.float64)
                        state[k].share_memory_(); assert state[k].is_shared()
                        # print("\t\t{}: {} convert to scalar tensor and moved to shared memory".format(k,state[k]))
                    else:
                        raise ValueError("Unknown optimizer-state type {}:{}".format(k,v))
            # 4) move optimzer.hyperparams to shared memory? No. They are constant (Bert, GPT2, Adam, SGD)
            # 5) clean up gradient

            for param in shared_model.parameters():
                param.grad = None
                assert param.grad is None, "cannot None grad?"

            gc.collect()
            # optimizer.zero_grad()
            self.shared_model = shared_model
            self.shared_optimizer = optimizer
            # print(f"self.shared_optimizer.param_groups:{self.shared_optimizer.param_groups}")
            # print(f"self.shared_optimizer.param_groups[0]:{self.shared_optimizer.param_groups[0]}")
            # exit(0)
            # print("[SharedOptimizer][id%d] sharing optimizer done."%self.id)


    @torch.no_grad()
    def init_in_subproc(self, id, cpu_layers, rank=-1, world_size=-1, no_pin_model=False, no_pin_grad_buf=False):
        """ Call this on entering subprocess """
        self.rank, self.world_size = rank, world_size
        self.no_pin_model = no_pin_model
        self.no_pin_grad_buf = no_pin_grad_buf


        if self.shared_optimizer is not None:
            # confirm model and optimizer are shared
            for param in self.shared_model.parameters():
                assert param.data.is_shared()
                assert param.requires_grad
                # assert param.grad is None, "{}".format(param.grad) # by default, gradient is process specific
            # print("[SharedOptimizer] rank{}'s model is shared".format(self.rank))
            for param, state in self.shared_optimizer.state.items(): # self.state = { param : {"step": 0, "exp_avg": tensor, "exp_avg_sq": tensor} } # per-param's optimization state (e.g, momentum_buffer, exp_avg, etc.).
                for k, v in state.items():
                    assert isinstance(v, torch.Tensor) and v.is_shared()


            for k, v in self.shared_optimizer.param_groups[0].items():
                if k != 'params': # 'lr': 0.001, 'betas': (0.9, 0.999), 'eps': 1e-08, 'weight_decay': 0.0005, 'amsgrad': False
                    assert (not isinstance(v, torch.Tensor)) # or (isinstance(v, torch.Tensor) and v.is_shared())
                # ref1: https://pytorch.org/docs/1.5.0/_modules/torch/optim/optimizer.html#Optimizer.state_dict
                # ref2: https://pytorch.org/docs/1.5.0/_modules/torch/optim/adam.html#Adam
            # print("[SharedOptimizer] rank{}'s optimizer is shared".format(self.rank))

            # initialize local pinned .grad # Trimed
            # for param in self.shared_model.parameters():
            #     assert param.requires_grad
            #     param.grad = torch.zeros(param.shape, dtype=param.dtype, device="cpu", requires_grad=False).pin_memory()
            #     assert not param.grad.is_shared() and param.grad.is_pinned()
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned .grad"%(self.id, self.rank))


        # for param in self.shared_model.parameters():
        #     param.ds_shape = param.shape
        # for name, buffer in self.shared_model.named_buffers():
        #     # print(f"Buffer name: {name}")
        #     # print(f"Buffer shape: {buffer.shape}")
        #     buffer.ds_shape = buffer.shape


        if self.no_pin_model:
            self.pinned_model = None

        elif id in cpu_layers:
             # initialize local pinned model (params and buffs)
            # with torch.no_grad():


            self.pinned_model = copy.deepcopy(self.shared_model)

            convert_to_pinned(self.pinned_model) # in-place convert a local model cpu to a pinned model
            # print("[SharedOptimizer][id%d] rank%d initialized local pinned model"%(self.id, self.rank))

        else:
            self.pinned_model = None


    def re_init(self):
        for param in self.shared_model.parameters():
            assert param.data.is_shared()
            assert param.requires_grad
        for param, state in self.shared_optimizer.state.items():
            for k, v in state.items():
                assert isinstance(v, torch.Tensor) and v.is_shared()

        self.pinned_model = copy.deepcopy(self.shared_model)
        convert_to_pinned(self.pinned_model)


    # shared_model.named_buffers()
    @torch.no_grad()
    def update_buf(self):
        """ In-place copy buffer from local pinned buf to shared model """
        if self.no_pin_grad_buf:
            return


        else:
            assert hasattr(self.shared_model, "pinned_buf")
            named_pin_buf = self.shared_model.pinned_buf
            for name, shared_buf in self.shared_model.named_buffers():
                if shared_buf is not None:
                    shared_buf.data.copy_(named_pin_buf[name].data, non_blocking=MEMCPY_NONBLK)
                else:
                    assert named_pin_buf[name] is None, "local pinned buffer must match shared model"

        # for shared_buf, pinned_buf in zip(self.shared_model.buffers(), self.pinned_model.buffers()):
        #     shared_buf.data.copy_(pinned_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from pinned memory to shared memory # nonblocking useless
        #     assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)

    @torch.no_grad()
    def step(self, zero_grad=False):
        if self.shared_optimizer is not None:
            self.shared_optimizer.step()
            if zero_grad:

                self.shared_optimizer.zero_grad()
        # confirm local .grad is still pinned
        # for param in self.shared_model.parameters():
        #     assert param.grad.is_pinned()
        # print("[SharedOptimizer] rank{} steped shared optimizer".format(self.rank))


    @torch.no_grad()
    def sync_pinned_model(self):
        """ In-place copy from shared model to local pinned model """
        if self.no_pin_model:
            return

        lines = [f"@@@@@@@@@@ rank:{self.rank}, layer:{self.id}: pinned model 参数:"]
        for name, param in self.shared_model.named_parameters():
            lines.append(f"  {name}: shape={param.shape}, {param}, dtype={param.dtype}, is_pinned={param.is_pinned()}")
        print("\n".join(lines), flush=True)
        #
        for pinned_param, shared_param in zip(self.pinned_model.parameters(), self.shared_model.parameters()):
            pinned_param.data.copy_(shared_param.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(self.pinned_model.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)
        # print("[SharedOptimizer] rank{} synced pinned model".format(self.rank))

    @torch.no_grad()
    def sync_pinned_buffer(self, pinned_buffer):
        for pinned_param, shared_param in zip(pinned_buffer.parameters(), self.shared_model.parameters()):
            print(f"rank:{self.rank}, pinned_param:{pinned_param}, shared_param:{shared_param}, is shared?{shared_param.is_shared()}")
            pinned_param.data.copy_(shared_param.view(-1).data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_param.is_pinned() and (not pinned_param.requires_grad) and shared_param.requires_grad
        for pinned_buf, shared_buf in zip(pinned_buffer.buffers(), self.shared_model.buffers()):
            pinned_buf.data.copy_(shared_buf.view(-1).data, non_blocking=MEMCPY_NONBLK) # inplace copy from shared memory to pinned memory  # nonblocking useless
            assert pinned_buf.is_pinned() and (not pinned_buf.requires_grad)


""" CPU update and sync model in background thread """
class UpdateInBkgd(object):
    """ Handles CPU update model in background thread for runtime.py
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task.
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, lr_scheduler, rank, nvprof=False, vt_profiler=None):
        self.swapout_stream = swapout_stream
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.rank = rank
        self.nvprof = nvprof
        self.vt_profiler = vt_profiler
        # initialize

        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread

        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread

        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True

        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        #
        self.the_last_put = None


    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task


            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers()))

            self._wait_swapout()


            self._update_buf(vt) # if using local pinned model for B'

            self._step(vt)

            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop()


    def _wait_swapout(self):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. # (maybe use cuda event in future)


    @torch.no_grad()
    def _update_buf(self, vt):
        """ update B of this pack """

        for l in vt.layers:


            self.shared_optimizer[l].update_buf()


    @torch.no_grad()
    def _step(self, vt):
        """ update W,K of this pack """
        for l in vt.layers:
            assert vt.In['dW'][l].medium == "LOC"
            assert vt.In['W'][l].medium == "SHM"
            assert vt.In['K'][l].medium == "SHM"
            assert vt.Out['W'][l].medium == "SHM"
            assert vt.Out['K'][l].medium == "SHM"
            start_time = time.time()

            self.shared_optimizer[l].step() # Update shared model and optim using swap-out'ed local .grad

            if self.lr_scheduler != []: # "gpt2_huggingface"
                if self.lr_scheduler[l] is not None:
                    assert self.shared_optimizer[l].shared_optimizer is not None
                    self.lr_scheduler[l].step()
                else:
                    assert self.shared_optimizer[l].shared_optimizer is None
            end_time = time.time()
            if self.vt_profiler is not None:
                self.vt_profiler.record_cpu_update(l, vt.idx, start_time, end_time)
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))


    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()


    def synchronize(self):
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break


class UpdateInBkgd_for_worker5_2(object):
    """ Handles CPU update model in background thread for runtime.py
        Assumption:
            0) simliar to FIFO queue. put each task, and get updated task.
            1) no sync_pinned_model(), which should be moved to next FWD/BWD's SwapIn
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, swapout_stream, shared_optimizer, lr_scheduler, layer_id_to_layer_idx, shared_model_nvme, layer_num, cpu_layer_id, rank, nvprof=False):
        self.swapout_stream = swapout_stream
        self.shared_optimizer = shared_optimizer
        self.lr_scheduler = lr_scheduler
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.shared_model_nvme: SharedModelNVMe_2 = shared_model_nvme
        self.rank = rank
        self.nvprof = nvprof
        # initialize

        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and update thread

        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between update and main thread

        self.update_thread = threading.Thread(target=self._thread_func)
        self.update_thread.daemon = True

        self.update_thread.start()
        # print("[UpdateInBkgd] rank{} started update_thread".format(self.rank))
        #
        self.the_last_put = None

        # 📍
        self.layer_num = layer_num
        self.cpu_layer_id = cpu_layer_id


    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task


            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) UPD(W)".format(vt.idx, vt.show_layers()))

            self._wait_swapout()


            self._update_buf(vt) # if using local pinned model for B'

            start_time = time.time()
            self._step(vt)
            end_time = time.time()
            step_time = end_time - start_time
            print(f"rank:{self.rank}, {vt.layers}更新完成, 耗时{step_time:.6f}秒", flush=True)
            # self._print_vt_pinned_params(vt)


            start_time = time.time()
            self._swap_out_from_pinned_buffer(vt)
            end_time = time.time()
            swap_out_time = end_time - start_time
            print(f"rank:{self.rank}, {vt.layers}从pinned buffer卸载完成, 耗时{swap_out_time:.6f}秒", flush=True)


            self.shared_model_nvme.delete_vts_shared_model_param_grad_buf(vt)


            self.get_queue.add((vt.idx, step_time, swap_out_time))
            if self.nvprof: nvtx_range_pop()


    def _wait_swapout(self):
        """ Wait swap out (dW,B') to complete (and reside in pinned memory) """
        self.swapout_stream.synchronize() # Wait for all the kernels in this stream to complete. # (maybe use cuda event in future)


    @torch.no_grad()
    def _update_buf(self, vt):
        """ update B of this pack """

        for l in vt.layers:


            self.shared_optimizer[l].update_buf()


    @torch.no_grad()
    def _step(self, vt):
        """ update W,K of this pack """
        for l in vt.layers:
            assert vt.In['dW'][l].medium == "LOC"
            assert vt.In['W'][l].medium == "SHM"
            assert vt.In['K'][l].medium == "SHM"
            assert vt.Out['W'][l].medium == "SHM"
            assert vt.Out['K'][l].medium == "SHM"

            self.shared_optimizer[l].step() # Update shared model and optim using swap-out'ed local .grad

            if self.lr_scheduler != []: # "gpt2_huggingface"
                if self.lr_scheduler[l] is not None:
                    assert self.shared_optimizer[l].shared_optimizer is not None
                    self.lr_scheduler[l].step()
                else:
                    assert self.shared_optimizer[l].shared_optimizer is None
        # print("[UpdateInBkgd] rank{} updated task{}({})".format(self.rank, vt.idx, vt.show_layers()))

    @torch.no_grad()
    def _print_vt_pinned_params(self, vt):
        """Print the pinned parameters for a virtual task."""
        for l in vt.layers:
            if vt.In['W'][l].medium == 'SHM' and vt.In['B'][l].medium == 'SHM':
                if l in self.cpu_layer_id:
                    pinned_model = self.shared_optimizer[l].pinned_model
                else:
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    pinned_model = self.shared_model_nvme.get_pinned_buffer(layer_idx)
            elif vt.In['W'][l].medium == 'PIN' and vt.In['B'][l].medium == 'PIN':
                if l in self.cpu_layer_id:
                    pinned_model = self.shared_optimizer[l].pinned_model
                else:
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    pinned_model = self.shared_model_nvme.get_pinned_buffer(layer_idx)
            else:
                raise ValueError("Underdevelopment")

            lines = [f"rank:{self.rank}, task:{vt.idx}, layer:{l} pinned memory 参数:"]
            for name, param in pinned_model.named_parameters():
                lines.append(
                    f"  {name}: shape={tuple(param.shape)}, dtype={param.dtype}, "
                    f"is_pinned={param.is_pinned()}, value={param}"
                )
            print("\n".join(lines), flush=True)


    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)
        self.the_last_put = vt.idx

    def get(self):
        ''' Call by main thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()

    #
    def synchronize(self):
        ''' Call by main thread. Blocking. Wait for all tasks in put_queue to complete. '''
        # depends on Assumption #0; wait for the last put idx
        if self.the_last_put is None:
            return
        # print("[UpdateInBkgd] rank{} synchronize until task{}".format(self.rank,self.the_last_put))
        while True:
            vt_idx, _, _ = self.get_queue.remove()
            if vt_idx == self.the_last_put:
                break
        # print("[UpdateInBkgd] rank{} has got task{}".format(self.rank,self.the_last_put))


    def _swap_out_from_pinned_buffer(self, vt):
        for layer_id in vt.layers:
            # if vt.has_data and layer_id == vt.layers[0]:
            #     continue
            # if layer_id == self.layer_num-2:
            #     continue
            # if layer_id == self.layer_num-3:
            #     continue

            # if vt.has_criterion and layer_id == vt.layers[-1]:
            #     continue
            if layer_id in self.cpu_layer_id:
                continue


            layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]

            print(f"rank:{self.rank}, {layer_id}准备从pinned buffer卸载", flush=True)
            self.shared_model_nvme.swap_out_from_pinned_buffer_sync(layer_id, layer_idx)
            print(f"rank:{self.rank}, {layer_id}从pinned buffer卸载完成", flush=True)


def delete_param_grad_buf_for_shared_model(top_module, manual_gc=False):
    @torch.no_grad()
    def fn(m): # m = each module
        for key, param in m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
            if param is not None:
                # delete param

                param.data = torch.empty(0, device="cpu")
                # delete grad

                if param.grad is not None:
                    param.grad = None


                # param.detach_() # in-place detaches self Tensor from the graph that created it, making it a leaf.
                # assert not param.requires_grad


        for key, buf in m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
            if buf is not None:
                m._buffers[key] = torch.empty(0, device="cpu")

    top_module.apply(fn) # Applies ``fn`` recursively to every submodule (as returned by ``.children()`` as well as self.
    #
    if manual_gc:
        gc.collect(); torch.cuda.empty_cache() # can block all cudaStreams


class SharedModelNVMe_2(object):

    def __init__(self, shared_model, empty_model, param_swapper, pinned_buffer_information, layer_count, transformer_layer_idx_to_shape, layer_id_to_layer_idx, layer_num, cpu_layer_id, rank):
        self.shared_model = shared_model
        self.param_swapper: AsyncPartitionedParameterSwapper = param_swapper
        self.pinned_buffer_information: PinnedBufferInformation_2 = pinned_buffer_information
        self.layer_count = layer_count
        self.transformer_layer_idx_to_shape = transformer_layer_idx_to_shape
        self.rank = rank


        self.pinned_buffer = []
        self.empty_model = empty_model


        for layer_idx in range(self.layer_count):
            self.initial_pinned_buffer(layer_idx)

        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.layer_num = layer_num
        self.cpu_layer_id = cpu_layer_id


    def initial_pinned_buffer(self, layer_idx):
        pinned_buffer = copy.deepcopy(self.empty_model[1])
        global_param_idx = 0
        for m in pinned_buffer.modules():
            for param_idx,(key, param) in enumerate(m._parameters.items()):
                compute_buffer = self.param_swapper._allocate_and_return_buffers_for_param(layer_idx, global_param_idx)
                param.data = compute_buffer.data
                global_param_idx+=1

            for key, buf in m._buffers.items():
                compute_buffer = self.param_swapper._allocate_and_return_buffers_for_param(layer_idx, global_param_idx)
                buf.data = compute_buffer.data
                global_param_idx += 1

        self.pinned_buffer.append(pinned_buffer)

    def get_pinned_buffer(self, layer_idx):
        return self.pinned_buffer[layer_idx]


    def swap_out_from_pinned_buffer(self, layer_id, layer_idx):
        # start_time = time.perf_counter()
        self.param_swapper.swap_out_transformer_layer(layer_id, layer_idx)
        # end_time = time.perf_counter()


    def swap_out_from_pinned_buffer_sync(self, layer_id, layer_idx):
        self.param_swapper.swap_out_transformer_layer_sync(layer_id, layer_idx)

    def _delete_param_grad_buf(self, model, manual_gc=False):
        delete_param_grad_buf_for_shared_model(model, manual_gc=manual_gc)

    def copy_shared_model_to_pinned_buffer(self, layer_id, layer_idx):
        pinned_buffer = self.get_pinned_buffer(layer_idx)
        global_param_idx = 0
        for cpu_m, pin_m in zip(self.shared_model[layer_id][0].modules(), pinned_buffer.modules()): # iterator over all modules in the network
            # print("{} <- {}".format(gpu_m, cpu_m))
            for key, param in cpu_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    # one_dim_param = param.view(-1)


                    pin_m._parameters[key].data.copy_(param.data.view(-1), non_blocking=MEMCPY_NONBLK)

            for key, buf in cpu_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    pin_m._buffers[key].data.copy_(buf.data.view(-1), non_blocking=MEMCPY_NONBLK)

    def swap_out_from_shared_memory_through_pinned_buffer(self, bwd_vts):
        for vt in bwd_vts:
            for layer_id in vt.layers:
                if layer_id in self.cpu_layer_id:
                    continue
                layer_idx = self.layer_id_to_layer_idx[vt.idx][layer_id]
                self.copy_shared_model_to_pinned_buffer(layer_id, layer_idx)
                self.swap_out_from_pinned_buffer_sync(layer_id, layer_idx)

                # for param in self.shared_model[layer_id][0].parameters():
                #     print(f"\t\trank:{self.rank} is shared?{param.is_shared()}")
                self._delete_param_grad_buf(self.shared_model[layer_id][0])

                # for param in self.shared_model[layer_id][0].parameters():
                #     print(f"\t\trank:{self.rank} is shared?{param.is_shared()}")


    def swap_in_from_nvme_to_shared_memory_through_pinned_buffer(self, bwd_vts_need_reload):
        for bwd_vt in bwd_vts_need_reload:
            for layer_id in bwd_vt.layers:
                if layer_id not in self.cpu_layer_id and layer_id in self.new_cpu_layer_id:
                    continue
                layer_idx = self.layer_id_to_layer_idx[bwd_vt.idx][layer_id]
                self.swap_in_sync(layer_id, layer_idx)
                self.alloc_shared_model_param_buf_and_reload_from_pinned_buffer(layer_id, layer_idx)


    def swap_in_from_nvme_to_shared_memory_through_pinned_buffer_without_bwdvt(self, new_cpu_layer_id):
        for i in range(0, len(new_cpu_layer_id), self.layer_count):

            # batch_layer_ids = new_cpu_layer_id[i:i+self.layer_count]
            batch_layer_ids = new_cpu_layer_id[i:min(i + self.layer_count, len(new_cpu_layer_id))]
            print(f"rank:{self.rank}, 从NVMe读取第{i//self.layer_count + 1}批layers: {batch_layer_ids}")


            layer_idx = 0
            for layer_id in batch_layer_ids:


                self.swap_in_sync(layer_id, layer_idx)

                self.alloc_shared_model_param_buf_and_reload_from_pinned_buffer(layer_id, layer_idx)
                layer_idx += 1


    def swap_in(self, layer_id, layer_idx): # reload_from_nvme_to_pinned_buffer
        self.param_swapper.swap_in_transformer_layer(layer_id, layer_idx)

    def swap_in_sync(self, layer_id, layer_idx):
        self.param_swapper.swap_in_transformer_layer_sync(layer_id, layer_idx)


    def make_shared_model_point_to_pinned(self, layer_id, layer_idx):

        pinned_buffer = self.get_pinned_buffer(layer_idx)

        global_param_idx = 0
        for cpu_m, pinned_m in zip(self.shared_model[layer_id][0].modules(), pinned_buffer.modules()):
            for key, param in pinned_m._parameters.items(): # OrderedDict([('weight', Parameter), ('bias', Parameter)])
                if param is not None:
                    cpu_m._parameters[key].data = param.view(self.transformer_layer_idx_to_shape[global_param_idx]).data
                    global_param_idx+=1
            for key, buf in pinned_m._buffers.items(): # OrderedDict([('running_mean', tensor), ('running_var', tensor), ('num_batches_tracked', tensor)])
                if buf is not None:
                    cpu_m._buffers[key].data = buf.view(self.transformer_layer_idx_to_shape[global_param_idx]).data
                    global_param_idx += 1

    def delete_vts_shared_model_param_grad_buf(self, vt):
        for layer_id in vt.layers:
            if layer_id in self.cpu_layer_id:
                continue
            self._delete_param_grad_buf(self.shared_model[layer_id][0])


    def delete_other_shared_model(self, other_layers):
        for layer_id in other_layers:
            if layer_id in self.cpu_layer_id:
                    continue
            self._delete_param_grad_buf(self.shared_model[layer_id][0])

    def delete_vts_shared_model_param_grad_buf_for_new_cpu_layer(self, vt, new_cpu_layer_id):
        for layer_id in vt.layers:
            if layer_id in self.cpu_layer_id or layer_id in new_cpu_layer_id:
                continue
            self._delete_param_grad_buf(self.shared_model[layer_id][0])


    @torch.no_grad()
    def alloc_shared_model_param_buf_and_reload_from_pinned_buffer(self, layer_id, layer_idx):
        """Allocate model parameter storage and restore it from the pinned buffer."""

        pinned_buffer_info = self.pinned_buffer_information


        pinned_buffer = self.get_pinned_buffer(layer_idx)

        shared_layer = self.shared_model[layer_id][0]

        print(f"rank:{self.rank}, 正在恢复{layer_id}")
        if layer_id == 5:
            for param in shared_layer.parameters():
                print(f"\t\trank:{self.rank} is shared?{param.is_shared()}")
        global_param_idx = 0
        for shared_m, pinned_m in zip(shared_layer.modules(), pinned_buffer.modules()):

            for key, param in pinned_m._parameters.items():
                if param is not None:

                    shared_param_data = torch.empty(
                        self.transformer_layer_idx_to_shape[global_param_idx],
                        device='cpu'
                    ).share_memory_()
                    shared_param_data.copy_(param.view(self.transformer_layer_idx_to_shape[global_param_idx]).data, non_blocking=MEMCPY_NONBLK)


                    shared_m._parameters[key].data = shared_param_data
                    if layer_id == 5:
                        print(f"\trank:{self.rank}, layer{layer_id}, shared_param.shape:{shared_m._parameters[key].shape}")
                    global_param_idx += 1


            for key, buf in pinned_m._buffers.items():
                if buf is not None:

                    shared_buf_data = torch.empty(
                        self.transformer_layer_idx_to_shape[global_param_idx],
                        device='cpu'
                    ).share_memory_()
                    shared_buf_data.copy_(buf.view(self.transformer_layer_idx_to_shape[global_param_idx]).data, non_blocking=MEMCPY_NONBLK)


                    shared_m._buffers[key].data = shared_buf_data
                    global_param_idx += 1

        # for m in shared_layer.modules():
        #     for key, param in m._parameters.items():
        #         if param is not None:

        #             param_shape = transformer_layer_shape[global_param_idx]
        #             param_numel = pinned_buffer_info.param_idx_in_layer_to_numel[global_param_idx]
        #             param_start_pos = pinned_buffer_info.param_idx_to_start_pos[global_param_idx]


        #             new_param_data = torch.empty(param_shape, device='cpu').share_memory_()


        #             pinned_data = pinned_buffer.data[param_start_pos:param_start_pos + param_numel]
        #             new_param_data.copy_(pinned_data.view(param_shape))


        #             shared_layer._parameters[key].data = new_param_data

        #             global_param_idx += 1

        #     for key, buf in m._buffers.items():
        #         if buf is not None:

        #             buf_shape = transformer_layer_shape[global_param_idx]
        #             buf_numel = pinned_buffer_info.param_idx_in_layer_to_numel[global_param_idx]
        #             buf_start_pos = pinned_buffer_info.param_idx_to_start_pos[global_param_idx]


        #             new_buf_data = torch.empty(buf_shape, device='cpu').share_memory_()


        #             pinned_buf_data = pinned_buffer.data[buf_start_pos:buf_start_pos + buf_numel]
        #             new_buf_data.copy_(pinned_buf_data.view(buf_shape))


        #             shared_layer._buffers[key].data = new_buf_data

        #             global_param_idx += 1

        print(f"rank{self.rank}: 成功从 pinned buffer 中恢复 layer {layer_id} 的参数和缓冲区。")
        if layer_id == 5:
            for param in shared_layer.parameters():
                print(f"\t\trank:{self.rank} is shared?{param.is_shared()}")


class SyncPinModelInBkgd(object):
    """ Handles synchronization to local pinned model in background thread.
        Assumption:
            0) always in FIFO ordering. put each task, and get synced task.
        TODO: skip sync_pinned_model() if already done (FWD -> BWD)
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.rank = rank
        self.nvprof = nvprof
        # initialize


        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread

        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between sync and prefetch thread
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        # print("[SyncPinModelInBkgd] rank{} started sync_thread".format(self.rank))


    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) SyncPin(W,B)".format(vt.idx, vt.show_layers()))


            self._sync_pinned_model(vt)

            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop()


    @torch.no_grad()
    def _sync_pinned_model(self, vt):
        """ sync W,B to local pinned model for this pack """
        for l in vt.layers:

            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                self.shared_optimizer[l].sync_pinned_model()

            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                pass
            else: # P2P
                raise ValueError("Underdevelopment")

    # self.put_queue.add(vt)
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)


    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()


from time import perf_counter as pc
class SyncPinModelInBkgd_for_worker5(object):
    """ Handles synchronization to local pinned model in background thread.
        Assumption:
            0) always in FIFO ordering. put each task, and get synced task.
        TODO: skip sync_pinned_model() if already done (FWD -> BWD)
        NOTE: move vt.medium check back to runtime by reducing granularity to vLayer?
    """
    def __init__(self, shared_optimizer, shared_model_nvme, layer_num, layer_id_to_layer_idx, cpu_layer_id, rank, nvprof=False):
        self.shared_optimizer = shared_optimizer
        self.rank = rank
        self.nvprof = nvprof
        # initialize


        self.put_queue = threadsafe_data_struct.Queue() # [vtask1, vtask2, ...] # between main and sync thread

        self.get_queue = threadsafe_data_struct.Queue() # [vtask1.idx, vtask2.idx, ...] # between sync and prefetch thread
        self.sync_thread = threading.Thread(target=self._thread_func)
        self.sync_thread.daemon = True
        self.sync_thread.start()
        # print("[SyncPinModelInBkgd] rank{} started sync_thread".format(self.rank))

        # 📍
        self.shared_model_nvme: SharedModelNVMe_2 = shared_model_nvme
        self.layer_num = layer_num
        self.layer_id_to_layer_idx = layer_id_to_layer_idx
        self.cpu_layer_id = cpu_layer_id


    def _thread_func(self):
        """ This method is to be executed from a daemon thread. """
        while True: # for each incoming task
            vt = self.put_queue.remove() # blocking
            if self.nvprof: nvtx_range_push("__task{}({}) SyncPin(W,B)".format(vt.idx, vt.show_layers()))


            self._sync_pinned_model(vt)

            self.get_queue.add(vt.idx)
            if self.nvprof: nvtx_range_pop()


    @torch.no_grad()
    def _sync_pinned_model(self, vt):
        """ sync W,B to local pinned model for this pack """
        print("rank:", self.rank, "@@@@@@@@@@正在进行shared memory/SSD到pinned memory的同步:", vt.show_layers(), flush=True)
        for l in vt.layers:

            if vt.In['W'][l].medium=='SHM' and vt.In['B'][l].medium=='SHM':
                if l in self.cpu_layer_id:
                    print(f"@@@@@@@@@@rank:{self.rank},layer{l} 是cpu layer", flush=True)
                    self.shared_optimizer[l].sync_pinned_model()
                else:
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    self.shared_model_nvme.swap_in_sync(l, layer_idx)

                    print(f"@@@@@@@@@@rank:{self.rank},layer{l} SWAP IN完成", flush=True)

                    if vt.type == 'BWD':
                        self.shared_model_nvme.make_shared_model_point_to_pinned(l, layer_idx)
                        print(f"rank:{self.rank},layer{l}的sharedmodel已指向pinned buffer", flush=True)

            # A reused PIN slot may contain another task's parameters; reload it.
            elif vt.In['W'][l].medium=='PIN' and vt.In['B'][l].medium=='PIN':
                if l in self.cpu_layer_id:
                    pass
                else:
                    layer_idx = self.layer_id_to_layer_idx[vt.idx][l]
                    self.shared_model_nvme.swap_in_sync(l, layer_idx)
                    if vt.type == 'BWD':
                        self.shared_model_nvme.make_shared_model_point_to_pinned(l, layer_idx)
            else: # P2P
                raise ValueError("Underdevelopment")

    # self.put_queue.add(vt)
    def iput(self, vt):
        ''' Call by main thread. Nonblocking.'''
        assert isinstance(vt, vTask)
        self.put_queue.add(vt)


    def get(self):
        ''' Call by prefetech thread. Blocking. Return vt.idx. '''
        return self.get_queue.remove()
