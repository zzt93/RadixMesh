import argparse
import logging
import sys
from time import sleep
from typing import Tuple

import torch

from src.test.correctness import args, test, prepare
from src.test.test_util import to_string, add_cli_args, generate_random_letter_list, generate_random_int_list, \
    CyclicBarrier
from src.util.log import configure_logger
from src.config.cache_config import load_server_args
from src.radix.radix_mesh import RadixMesh
from src.radix.core_enum import RadixMode

import random
import string
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)


def random_write(idx_arg: Tuple[int, str], latch: CyclicBarrier):
    idx, arg = idx_arg
    server_args = prepare(arg)
    radix_mesh = RadixMesh(server_args)
    if radix_mesh.mode != RadixMode.ROUTER:
        for _ in range(10):
            key = generate_random_letter_list()
            radix_mesh.insert(key, torch.tensor(generate_random_int_list(len(key))))


if __name__ == '__main__':
    test(random_write)
