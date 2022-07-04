import os
import os.path as osp
import argparse
import json
from pathlib import Path
import time
import psutil
import torch

import logging
from tensorboardX import SummaryWriter


def parse_args():
    parser = argparse.ArgumentParser("Train the unsupervised human pose estimation network")
    parser.add_argument('--test_name', help='Specify the test name', default='gpu')
    parser.add_argument('--output_dir', help="Specify the directory of data", default='logs')
    parser.add_argument('--log_dir', help='Specify the directory of output', default='logs')

    parser.add_argument('--gpu', type=int, help="Specify the gpu to monitor", default=-1)
    parser.add_argument('--freq', type=int, help="Specify the monitor frequency, unit hz", default=10)
    parser.add_argument('--time', type=int, help="Specify how long to monitor, unit s", default=60)
    args = parser.parse_args()
    return args


def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(f'gpu_id {gpu_id} 对应的显卡不存在!')
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free


def get_cpu_mem_info():
    """
    获取当前机器的内存信息, 单位 MB
    :return: mem_total 当前机器所有的内存 mem_free 当前机器可用的内存 mem_process_used 当前进程使用的内存
    """
    mem_total = round(psutil.virtual_memory().total / 1024 / 1024, 2)
    mem_free = round(psutil.virtual_memory().available / 1024 / 1024, 2)
    mem_process_used = round(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, 2)
    return mem_total, mem_free, mem_process_used


def create_logger(args):
    root_output_dir = Path(args.output_dir)
    test_name = args.test_name
    if not root_output_dir.exists():
        print(f'=> creating {root_output_dir}')
        root_output_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = root_output_dir / test_name
    final_output_dir.mkdir(parents=True, exist_ok=True)
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}.log'.format(test_name, time_str)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(args.log_dir) / (test_name + "_" + time_str)
    print(f"=> creating {tensorboard_log_dir}")
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
    return logger, str(final_output_dir), str(tensorboard_log_dir)


if __name__ == "__main__":

    args = parse_args()
    logger, final_output_dir, tb_log_dir = create_logger(args)
    summary_writer = SummaryWriter(log_dir=tb_log_dir)

    for count in range(args.time * args.freq):
        if args.gpu == -1:
            gpu_list = list(range(torch.cuda.device_count()))
        else:
            gpu_list = [args.gpu]
        for gpu_id in gpu_list:
            gpu_mem_total, gpu_mem_used, gpu_mem_free = get_gpu_mem_info(gpu_id)
            summary_writer.add_scalar(f"gpu-mem-{gpu_id}/total", gpu_mem_total, count)
            summary_writer.add_scalar(f"gpu-mem-{gpu_id}/used", gpu_mem_used, count)
            summary_writer.add_scalar(f"gpu-mem-{gpu_id}/free", gpu_mem_free, count)
            gpu_mem_info = {
                "gpu_id": gpu_id,
                "gpu_mem_total": gpu_mem_total,
                "gpu_mem_used": gpu_mem_used,
                "gpu_mem_free": gpu_mem_free
            }
            logger.info(json.dumps(gpu_mem_info))

        cpu_mem_total, cpu_mem_free, cpu_mem_process_used = get_cpu_mem_info()
        summary_writer.add_scalar(f"cpu-mem/total", cpu_mem_total, count)
        summary_writer.add_scalar(f"cpu-mem/used", cpu_mem_free, count)
        summary_writer.add_scalar(f"cpu-mem/free", cpu_mem_process_used, count)
        cpu_mem_info = {
                "cpu_mem_total": cpu_mem_total,
                "cpu_mem_free": cpu_mem_free,
                "cpu_mem_process_used": cpu_mem_process_used
            }
        logger.info(json.dumps(cpu_mem_info))

        time.sleep(1 / args.freq)
