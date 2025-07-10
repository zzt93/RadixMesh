import argparse
import random
import string
from multiprocessing.managers import SyncManager

from src.radix.sglang.srt.mem_cache.radix_cache import TreeNode
from multiprocessing import Process, Value, Condition


def to_string(node: TreeNode, level=0) -> str:
    indent = '  ' * level
    s = f"{indent}TreeNode(id={node.id}, hit_count={node.hit_count}, key={node.key})"
    return s


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


class CountDownLatch:
    def __init__(self, count, cond):
        self.count = count
        self.condition = cond

    def count_down(self):
        with self.condition:
            self.count.value -= 1
            if self.count.value == 0:
                self.condition.notify_all()

    def wait(self):
        with self.condition:
            while self.count.value > 0:
                self.condition.wait()
