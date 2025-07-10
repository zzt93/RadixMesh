import argparse
import logging
import sys
from time import sleep

import torch

from src.test.test_util import to_string, add_cli_args, generate_random_letter_list, generate_random_int_list
from src.util.log import configure_logger
from src.config.cache_config import load_server_args
from src.radix.radix_mesh import RadixMesh
from src.radix.core_enum import RadixMode

import random
import string
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


def process_one_config(arg: str):
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    raw_args = parser.parse_args(arg.split())  # 注意 split
    server_args = load_server_args(raw_args.config_file)
    node_rank = server_args.prefill_node_rank if server_args.prefill_node_rank != -1 else server_args.decode_node_rank if server_args.decode_node_rank != -1 else server_args.router_node_rank
    configure_logger(f"node@{node_rank}")

    radix_mesh = RadixMesh(server_args)
    if radix_mesh.mode != RadixMode.ROUTER:
        for _ in range(10):
            key = generate_random_letter_list()
            radix_mesh.insert(key, torch.tensor(generate_random_int_list(len(key))))

    # radix_mesh.pretty_print()


def test():
    args = [
        '--config-file src/test/p1.yaml',
        '--config-file src/test/p2.yaml',
        '--config-file src/test/p3.yaml',
        '--config-file src/test/d1.yaml',
        '--config-file src/test/r1.yaml'
    ]

    with ProcessPoolExecutor() as executor:
        # 提交任务到进程池
        list(executor.map(process_one_config, args))


if __name__ == '__main__':
    test()
