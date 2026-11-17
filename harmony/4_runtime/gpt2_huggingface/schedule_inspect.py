"""Inspect the repository's schedule pickles without importing or executing globals."""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import pickletools


@dataclass
class _Global:
    module: str
    name: str


@dataclass
class _Instance:
    kind: _Global
    state: dict = field(default_factory=dict)


def read_schedule(path):
    """Decode data containers and passive task records; never invoke pickle callables."""
    data = Path(path).read_bytes()
    if len(data) > 10 * 1024 * 1024:
        raise ValueError('Schedule exceeds the inspection size limit')
    stack, memo = [], {}
    marker = object()

    def marked():
        index = next(i for i in range(len(stack) - 1, -1, -1) if stack[i] is marker)
        values = stack[index + 1:]
        del stack[index:]
        return values

    for opcode, arg, _ in pickletools.genops(data):
        op = opcode.name
        if op in ('PROTO', 'FRAME'):
            continue
        if op == 'MARK':
            stack.append(marker)
        elif op in ('BININT', 'BININT1', 'BININT2', 'INT', 'LONG', 'LONG1', 'LONG4',
                    'UNICODE', 'SHORT_BINUNICODE', 'BINUNICODE', 'BINUNICODE8', 'BINFLOAT', 'FLOAT'):
            stack.append(arg)
        elif op == 'NONE':
            stack.append(None)
        elif op in ('NEWTRUE', 'NEWFALSE'):
            stack.append(op == 'NEWTRUE')
        elif op == 'EMPTY_LIST':
            stack.append([])
        elif op == 'EMPTY_DICT':
            stack.append({})
        elif op == 'EMPTY_TUPLE':
            stack.append(())
        elif op in ('BINPUT', 'LONG_BINPUT', 'PUT'):
            memo[int(arg)] = stack[-1]
        elif op == 'MEMOIZE':
            memo[len(memo)] = stack[-1]
        elif op in ('BINGET', 'LONG_BINGET', 'GET'):
            stack.append(memo[int(arg)])
        elif op == 'APPEND':
            value = stack.pop()
            stack[-1].append(value)
        elif op == 'APPENDS':
            values = marked()
            stack[-1].extend(values)
        elif op == 'SETITEM':
            value, key = stack.pop(), stack.pop()
            stack[-1][key] = value
        elif op == 'SETITEMS':
            values = marked()
            stack[-1].update(zip(values[::2], values[1::2]))
        elif op == 'TUPLE':
            stack.append(tuple(marked()))
        elif op in ('TUPLE1', 'TUPLE2', 'TUPLE3'):
            count = int(op[-1])
            values = tuple(stack[-count:])
            del stack[-count:]
            stack.append(values)
        elif op in ('STACK_GLOBAL', 'GLOBAL'):
            if op == 'STACK_GLOBAL':
                name, module = stack.pop(), stack.pop()
            else:
                module, name = arg.split(' ', 1)
            if (module, name) not in {('collections', 'OrderedDict'), ('task_data_struct', 'vTask'), ('task_data_struct', 'Medium')}:
                raise ValueError('Unsupported serialized global: ' + module + '.' + name)
            stack.append(_Global(module, name))
        elif op == 'REDUCE':
            args, symbol = stack.pop(), stack.pop()
            if symbol != _Global('collections', 'OrderedDict') or args != ():
                raise ValueError('Only an empty serialized OrderedDict is supported')
            stack.append({})
        elif op == 'NEWOBJ':
            args, symbol = stack.pop(), stack.pop()
            if not isinstance(symbol, _Global) or symbol.module != 'task_data_struct' or args != ():
                raise ValueError('Unsupported passive task record')
            stack.append(_Instance(symbol))
        elif op == 'BUILD':
            state = stack.pop()
            if not isinstance(stack[-1], _Instance) or not isinstance(state, dict):
                raise ValueError('Unsupported serialized task state')
            stack[-1].state.update(state)
        elif op == 'STOP':
            if len(stack) != 1:
                raise ValueError('Invalid schedule stack')
            rtasks, configs = stack[0]
            if not isinstance(rtasks, dict) or not isinstance(configs, dict):
                raise ValueError('Invalid schedule containers')
            return rtasks, configs, hashlib.sha256(data).hexdigest()
        else:
            raise ValueError('Unsupported pickle opcode: ' + op)
    raise ValueError('Incomplete schedule')


def inspect_schedule(path, require_microbatch_one=False):
    rtasks, configs, checksum = read_schedule(path)
    sizes = set()
    task_count = 0
    for rank, tasks in rtasks.items():
        for task in tasks:
            state = task.state
            if state['type'] in ('FWD', 'BWD'):
                sizes.update(state['ubatchszs'])
                task_count += 1
    if require_microbatch_one:
        def flatten(value):
            for item in value:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item
        if (configs['u_fwd'], configs['u_bwd']) != (1, 1) or sizes != {1}:
            raise ValueError('Capacity experiments require every forward/backward task to use microbatch size 1')
        if any(v != 1 for name in ('ubatchszs_fwd', 'ubatchszs_bwd') for v in flatten(configs[name])):
            raise ValueError('Capacity configuration contains a microbatch size other than 1')
    return {'path': str(Path(path).resolve()), 'sha256': checksum,
            'mode': configs['mode'], 'N': configs['N'], 'D': configs['D'], 'R': configs['R'],
            'u_fwd': configs['u_fwd'], 'u_bwd': configs['u_bwd'],
            'max_pack_size': max(len(pack) for name in ('pack_fwd', 'pack_bwd') for pack in configs[name]),
            'task_microbatch_sizes': sorted(sizes), 'compute_tasks': task_count}
