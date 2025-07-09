import argparse
import logging
import sys

import torch

from src.util.log import configure_logger
from src.config.cache_config import load_server_args
from src.radix.radix_mesh import RadixMesh, RadixMode

import random
import string
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


def add_cli_args(parser: argparse.ArgumentParser):
    # Model and port args
    parser.add_argument(
        "--config-file",
        type=str,
        help="The path of the cache nodes",
        required=True,
    )


def generate_random_letter_list(min_len=1, max_len=20):
    length = random.randint(min_len, max_len)
    return [random.choice(string.ascii_lowercase) for _ in range(length)]


def generate_random_int_list(length, min_value=1, max_value=1000):
    return [random.randint(min_value, max_value) for _ in range(length)]


def process_one_config(arg: str):
    configure_logger()

    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    raw_args = parser.parse_args(arg.split())  # 注意 split
    server_args = load_server_args(raw_args.config_file)
    # print(server_args)

    radix_mesh = RadixMesh(server_args)
    if radix_mesh.mode != RadixMode.ROUTER:
        radix_mesh.insert([1, 2, 3], torch.tensor([10, 20, 30]))
        for _ in range(10):
            key = generate_random_letter_list()
            radix_mesh.insert(key, torch.tensor(generate_random_int_list(len(key))))

    # logger.info(f"Radix mesh: {radix_mesh}")
    # radix_mesh.pretty_print()
    res = radix_mesh.match_prefix([1, 2, 3])
    logger.info(f"{res}")


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
