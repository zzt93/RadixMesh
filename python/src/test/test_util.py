import argparse
import random
import string
from multiprocessing.managers import SyncManager

from src.radix.sglang.srt.mem_cache.radix_cache import TreeNode
from multiprocessing import Process, Value, Condition


def to_string(node: TreeNode, level=0) -> str:
    indent = '  ' * level
    s = f"{indent}TreeNode(id={node.id}, hit_count={node.hit_count}, key={node.key}, value={node.value})"
    return s


def add_cli_args(parser: argparse.ArgumentParser):
    # Model and port args
    parser.add_argument(
        "--config-file",
        type=str,
        help="The path of the cache nodes",
        required=True,
    )


def generate_random_letter_list(min_len=1, max_len=20, min_value=100, max_value=200):
    length = random.randint(min_len, max_len)
    return [random.randint(min_value, max_value) for _ in range(length)]


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


class CyclicBarrier:
    def __init__(self, count, manager, cond=None):
        self.parties = count  # 初始计数（屏障的参与方数量）
        self.count = manager.Value('i', count)  # 当前剩余计数
        self.generation = manager.Value('i', 0)  # "代"计数器，用于区分重置周期
        if cond is None:
            cond = manager.Condition()
        self.condition = cond

    def wait(self):
        with self.condition:
            gen = self.generation.value  # 记录进入屏障时的"代"
            self.count.value -= 1  # 减少当前计数

            if self.count.value == 0:  # 如果是最后一个到达的线程
                # 重置屏障：恢复计数并更新"代"
                self.count.value = self.parties
                self.generation.value += 1
                self.condition.notify_all()  # 唤醒所有等待线程
            else:
                # 等待直到"代"发生变化（表示屏障已重置）
                while gen == self.generation.value:
                    self.condition.wait()
