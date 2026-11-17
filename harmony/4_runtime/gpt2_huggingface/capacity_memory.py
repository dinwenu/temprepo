"""Inspect memory constraints for Section 4.2; never change host policy."""
from pathlib import Path
import re


def node_ids(value):
    if not re.fullmatch(r'\d+(,\d+)*', value):
        raise ValueError('NUMA nodes must be comma-separated nonnegative IDs, e.g. 0 or 0,1')
    return sorted(set(int(item) for item in value.split(',')))


def cgroup_limits():
    """Read effective cgroup-v2 limits, including inherited ancestor limits."""
    mount = Path('/sys/fs/cgroup')
    entry = next((line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')), None)
    if entry is None:
        return {'memory_max_bytes': None, 'swap_disabled': False, 'paths': []}
    group = mount / entry.lstrip('/')
    if not group.exists():
        raise ValueError('Cannot locate the current cgroup-v2 directory')
    values, swap, paths = [], [], []
    while True:
        memory, swapfile = group / 'memory.max', group / 'memory.swap.max'
        if memory.is_file():
            value = memory.read_text().strip()
            if value != 'max':
                values.append(int(value))
            paths.append(str(memory))
        if swapfile.is_file():
            swap.append(swapfile.read_text().strip())
        if group == mount:
            break
        if mount not in group.parents:
            raise ValueError('Unexpected cgroup path')
        group = group.parent
    return {'memory_max_bytes': min(values) if values else None, 'swap_disabled': '0' in swap, 'paths': paths}


def inspect_capacity(memory_gb, source, cpu_nodes, memory_nodes):
    expected = memory_gb * 1024 ** 3
    if memory_gb == 256:
        if cpu_nodes or memory_nodes or source != 'numa':
            raise ValueError('NUMA/cgroup binding options apply only to the 128 GB experiment; the 256 GB experiment runs directly on the full-memory server.')
        text = Path('/proc/meminfo').read_text()
        total = int(re.search(r'MemTotal:\s+(\d+)\s+kB', text).group(1)) * 1024
        cgroup = cgroup_limits()
        errors = []
        if not 0.95 * expected <= total <= 1.05 * expected:
            errors.append(f'The full-memory experiment requires the 256 GiB reference server; this host has {total / 1024 ** 3:.2f} GiB.')
        if cgroup['memory_max_bytes'] is not None and cgroup['memory_max_bytes'] < 0.95 * expected:
            errors.append('An inherited cgroup memory limit is smaller than the full-memory experiment budget.')
        return {'requested_gib': memory_gb, 'limit_source': 'system',
                'system_total_bytes': total, 'cgroup': cgroup, 'errors': errors}
    allowed = node_ids(memory_nodes)
    for node in node_ids(cpu_nodes):
        if not Path(f'/sys/devices/system/node/node{node}').is_dir():
            raise ValueError(f'CPU NUMA node {node} is not present')
    bound = 0
    for node in allowed:
        text = Path(f'/sys/devices/system/node/node{node}/meminfo').read_text()
        match = re.search(r'MemTotal:\s+(\d+)\s+kB', text)
        if not match:
            raise ValueError(f'Cannot read memory capacity for NUMA node {node}')
        bound += int(match.group(1)) * 1024
    cgroup = cgroup_limits()
    errors = []
    if source == 'numa':
        if not 0.95 * expected <= bound <= 1.05 * expected:
            errors.append(f'Bound NUMA nodes contain {bound / 1024 ** 3:.2f} GiB, not approximately {memory_gb} GiB; choose matching nodes or use an explicit cgroup limit.')
        if cgroup['memory_max_bytes'] is not None and cgroup['memory_max_bytes'] < 0.95 * expected:
            errors.append('An inherited cgroup memory limit is smaller than the requested experiment budget.')
        swap_total = int(re.search(r'SwapTotal:\s+(\d+)', Path('/proc/meminfo').read_text()).group(1))
        if swap_total and not cgroup['swap_disabled']:
            errors.append('Swap is enabled; use a no-swap cgroup or a reference machine with swap disabled for the memory-capacity comparison.')
    else:
        limit = cgroup['memory_max_bytes']
        if limit is None or abs(limit - expected) > 1024 ** 2:
            errors.append(f'The inherited cgroup-v2 memory.max must be {memory_gb} GiB for this experiment.')
        if not cgroup['swap_disabled']:
            errors.append('The inherited cgroup-v2 memory.swap.max must be 0.')
        if bound < 0.95 * expected:
            errors.append('The NUMA memory binding exposes less physical memory than the requested cgroup budget.')
    return {'requested_gib': memory_gb, 'limit_source': source, 'cpu_nodes': cpu_nodes,
            'memory_nodes': memory_nodes, 'bound_numa_bytes': bound, 'cgroup': cgroup, 'errors': errors}
