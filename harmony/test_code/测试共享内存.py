import torch
import multiprocessing as mp

def f(shared_tensor):
    # 在进程内修改共享张量的数据
    shared_tensor[0] += 1

if __name__ == '__main__':
    # 创建一个共享张量
    shared_tensor = torch.zeros(1)
    shared_tensor.share_memory_()  # 将张量数据移动到共享内存中

    # 创建多个进程来修改共享张量的数据
    processes = []
    for _ in range(4):
        p = mp.Process(target=f, args=(shared_tensor,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Final value of shared_tensor:", shared_tensor[0])

