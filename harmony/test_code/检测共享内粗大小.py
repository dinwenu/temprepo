import psutil
from psutil._common import bytes2human

def get_shared_memory_size():
    mem = psutil.virtual_memory()
    shared_memory_size = mem.shared
    return shared_memory_size

# if __name__ == "__main__":
#     shared_memory_size = get_shared_memory_size()
#     s = psutil.virtual_memory()
#     value = getattr(s, "shared")
#     print("Shared memory size:", bytes2human(shared_memory_size))
#     print("Shared memory size:", bytes2human(value))

def get_shared_memory_info():
    # 获取共享内存信息
    shared_memory = psutil.virtual_memory().shared
    
    # 获取系统总内存大小
    total_memory = psutil.virtual_memory().total
    
    # 计算剩余内存大小
    free_memory = total_memory - shared_memory
    
    # 返回共享内存信息
    return {
        "total": total_memory,
        "used": shared_memory,
        "free": free_memory
    }

# 获取共享内存信息
shared_memory_info = get_shared_memory_info()

# 打印共享内存信息
print("共享内存总大小: {}".format(bytes2human(shared_memory_info["total"])))
print("已使用的共享内存大小: {}".format(bytes2human(shared_memory_info["used"])))
print("剩余的共享内存大小: {}".format(bytes2human(shared_memory_info["free"])))