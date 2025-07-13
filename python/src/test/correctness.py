import argparse
import logging
import multiprocessing
from functools import partial
from time import sleep
from typing import Callable, Tuple

import torch

from src.router.cache_aware_router import CacheAwareRouter
from src.test.test_util import to_string, add_cli_args, generate_random_letter_list, generate_random_int_list, \
    CountDownLatch, CyclicBarrier
from src.util.log import configure_logger
from src.config.cache_config import load_server_args
from src.radix.radix_mesh import RadixMesh
from src.radix.core_enum import RadixMode

from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

args = [
    '--config-file src/test/p1.yaml',
    '--config-file src/test/p2.yaml',
    '--config-file src/test/p3.yaml',
    '--config-file src/test/d1.yaml',
    '--config-file src/test/d2.yaml',
    '--config-file src/test/r1.yaml'
]


def sync_and_routing(idx_arg: Tuple[int, str], latch: CyclicBarrier):
    idx, arg = idx_arg
    server_args = prepare(arg)
    radix_mesh = RadixMesh(server_args)
    if radix_mesh.mode == RadixMode.ROUTER:
        router = CacheAwareRouter(radix_mesh, skip_warm_up=True)

    # TEST prefill insert
    logger.info("=====TEST prefill insert=====")
    insert_node_rank = 1
    k1 = [1, 2, 3]
    v1 = torch.tensor([10, 20, 30])
    if radix_mesh.mode == RadixMode.PREFILL:
        if radix_mesh.global_node_rank() == insert_node_rank:
            radix_mesh.insert(k1, v1)
        for _ in range(10):
            key = generate_random_letter_list()
            radix_mesh.insert(key, torch.tensor(generate_random_int_list(len(key))))

    sleep(1)
    res = radix_mesh.match_prefix(k1)
    if radix_mesh.mode == RadixMode.PREFILL:
        logger.info(f"test result of PREFILL node: {res.device_indices}, {to_string(res.last_device_node)}")
        assert torch.equal(res.device_indices, v1)
    elif radix_mesh.mode == RadixMode.ROUTER:
        logger.info(f"test result of ROUTER node: {res.prefill_node_rank}")
        assert res.prefill_node_rank == insert_node_rank, f"expect {insert_node_rank}, get {res.prefill_node_rank}"
    res = radix_mesh.match_prefix([1, 2, 3, 4])
    if radix_mesh.mode == RadixMode.PREFILL:
        logger.info(f"test result of PREFILL node: {res.device_indices}, {to_string(res.last_device_node)}")
        assert torch.equal(res.device_indices, v1)
    elif radix_mesh.mode == RadixMode.ROUTER:
        logger.info(f"test result of ROUTER node: {res.prefill_node_rank}, decode rank {res.decode_node_rank}")
        assert res.prefill_node_rank == insert_node_rank, f"expect {insert_node_rank}, get {res.prefill_node_rank}"
        assert res.decode_node_rank == -1, f"expect -1, get {res.decode_node_rank}"
        r = router.cache_aware_route([1, 2, 3, 4])
        logger.info(f"{r}")
        assert r.prefill_addr == server_args.prefill_cache_nodes[
            insert_node_rank], f"expect {server_args.prefill_cache_nodes[insert_node_rank]}, get {r.prefill_addr}"

    logger.info("=====TEST prefill insert done =====")
    latch.wait()

    # TEST decode insert
    logger.info("=====TEST decode insert=====")
    decode_insert_node_rank = len(server_args.prefill_cache_nodes)
    assert server_args.is_decode_node_rank(decode_insert_node_rank)
    k1 = [1, 2, 3, 4, 5, 6]
    v1 = torch.tensor([10, 20, 30, 40, 50, 60])
    if radix_mesh.global_node_rank() == decode_insert_node_rank:
        radix_mesh.insert(k1, v1)

    sleep(1)
    res = radix_mesh.match_prefix(k1)
    if radix_mesh.mode == RadixMode.PREFILL or radix_mesh.mode == RadixMode.DECODE:
        logger.info(f"test result of PREFILL node: {res.device_indices}, {to_string(res.last_device_node)}")
        assert torch.equal(res.device_indices, v1)
    elif radix_mesh.mode == RadixMode.ROUTER:
        logger.info(f"test result of ROUTER node: {res.prefill_node_rank}")
        assert res.prefill_node_rank == insert_node_rank, f"expect {insert_node_rank}, get {res.prefill_node_rank}"
        assert res.decode_node_rank == decode_insert_node_rank, f"expect {decode_insert_node_rank}, get {res.decode_node_rank}"
    res = radix_mesh.match_prefix([1, 2, 3, 4, 5, 6, 7])
    if radix_mesh.mode == RadixMode.PREFILL or radix_mesh.mode == RadixMode.DECODE:
        logger.info(f"test result of PREFILL node: {res.device_indices}, {to_string(res.last_device_node)}")
        assert torch.equal(res.device_indices, v1)
    elif radix_mesh.mode == RadixMode.ROUTER:
        logger.info(f"test result of ROUTER node: {res.prefill_node_rank}, {res.decode_node_rank}")
        assert res.prefill_node_rank == insert_node_rank, f"expect {insert_node_rank}, get {res.prefill_node_rank}"
        assert res.decode_node_rank == decode_insert_node_rank, f"expect {decode_insert_node_rank}, get {res.decode_node_rank}"
        r = router.cache_aware_route([1, 2, 3, 4, 5, 6, 7])
        assert r.prefill_addr == server_args.prefill_cache_nodes[insert_node_rank]
        assert r.decode_addr == server_args.decode_cache_nodes[server_args.local_node_rank(decode_insert_node_rank)]


def prepare(arg):
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    raw_args = parser.parse_args(arg.split())  # 注意 split
    server_args = load_server_args(raw_args.config_file)
    node_rank = server_args.prefill_node_rank if server_args.prefill_node_rank != -1 else server_args.decode_node_rank if server_args.decode_node_rank != -1 else server_args.router_node_rank
    configure_logger(f"node@{node_rank}")
    return server_args


def _fn_wrapper(item, fn, latch, logger):
    try:
        result = fn(item, latch=latch)
        return True, result
    except Exception as e:
        logger.exception(f"test exception: %s", e)
        return False, str(e)


def test(fn: Callable, count: int = len(args)):
    with multiprocessing.Manager() as m:
        count_initial = count
        cond = m.Condition()
        barrier = CyclicBarrier(count_initial, m, cond)  # 使用CyclicBarrier

        with ProcessPoolExecutor() as executor:
            # 提交任务到进程池
            fc = partial(_fn_wrapper, fn=fn, latch=barrier, logger=logger)
            list(executor.map(fc, enumerate(args)))


def multi_write(idx_arg: Tuple[int, str], latch: CyclicBarrier):
    idx, arg = idx_arg
    server_args = prepare(arg)

    radix_mesh = RadixMesh(server_args)
    if radix_mesh.mode == RadixMode.ROUTER:
        router = CacheAwareRouter(radix_mesh)

    # TEST prefill insert
    logger.info("=====TEST multi write insert=====")
    k1 = [1, 2, 3]
    v1 = torch.tensor([10, 20, 30])
    if radix_mesh.mode == RadixMode.PREFILL:
        if radix_mesh.global_node_rank() == idx:
            v1_tmp = v1.clone()
            v1_tmp[-1] += idx
            logger.info(f"{v1_tmp}")
            radix_mesh.insert(k1, v1_tmp)

    latch.wait()
    sleep(1)
    res = radix_mesh.match_prefix(k1)
    if radix_mesh.mode == RadixMode.PREFILL or radix_mesh.mode == RadixMode.DECODE:
        logger.info(f"test result of PREFILL node: {to_string(res.last_device_node)}, {res.last_device_node.value.value}")
        assert torch.equal(res.device_indices, v1), f"expect {v1}, get {res.device_indices}"
    elif radix_mesh.mode == RadixMode.ROUTER:
        logger.info(f"test result of ROUTER node: {res.prefill_node_rank}")
        assert res.prefill_node_rank == radix_mesh.sync_algo.master_node_rank(), f"expect {radix_mesh.sync_algo.master_node_rank()}, get {res.prefill_node_rank}"
    res = radix_mesh.match_prefix([1, 2, 3, 4])
    if radix_mesh.mode == RadixMode.PREFILL or radix_mesh.mode == RadixMode.DECODE:
        logger.info(f"test result of PREFILL node: {to_string(res.last_device_node)}, {res.last_device_node.value.value}")
        assert torch.equal(res.device_indices, v1), f"expect {v1}, get {res.device_indices}"
    elif radix_mesh.mode == RadixMode.ROUTER:
        logger.info(f"test result of ROUTER node: {res.prefill_node_rank}")
        assert res.prefill_node_rank == radix_mesh.sync_algo.master_node_rank(), f"expect {radix_mesh.sync_algo.master_node_rank()}, get {res.prefill_node_rank}"
        assert res.decode_node_rank == -1, f"expect -1, get {res.decode_node_rank}"
        r = router.cache_aware_route([1, 2, 3, 4])
        assert r.prefill_addr == server_args.prefill_cache_nodes[radix_mesh.sync_algo.master_node_rank()], f"expected {server_args.prefill_cache_nodes[radix_mesh.sync_algo.master_node_rank()]} get {r.prefill_addr}"

    latch.wait()
    logger.info("=====TEST multi write insert2=====")
    k2 = [1, 2, 3, 4, 5, 6, 7]
    v2 = torch.tensor([10, 20, 30, 40, 50, 60, 70])
    if radix_mesh.mode == RadixMode.PREFILL:
        if radix_mesh.global_node_rank() == 2:
            k2_tmp = k2
            v2_tmp = v2.clone()
            logger.info(f"{v2_tmp}")
            radix_mesh.insert(k2_tmp, v2_tmp)
        if radix_mesh.global_node_rank() == 1:
            k2_tmp = k2[:-1]
            v2_tmp = v2.clone()[:-1]
            logger.info(f"{v2_tmp}")
            radix_mesh.insert(k2_tmp, v2_tmp)
        if radix_mesh.global_node_rank() == 0:
            k2_tmp = k2[:-2]
            v2_tmp = v2.clone()[:-2]
            logger.info(f"{v2_tmp}")
            radix_mesh.insert(k2_tmp, v2_tmp)

    latch.wait()
    sleep(1)
    res = radix_mesh.match_prefix(k2)
    if radix_mesh.mode == RadixMode.PREFILL or radix_mesh.mode == RadixMode.DECODE:
        logger.info(f"test result of PREFILL node: {to_string(res.last_device_node)}, {res.last_device_node.value.value}")
        assert torch.equal(res.device_indices, v2), f"expect {v2}, get {res.device_indices}"
    elif radix_mesh.mode == RadixMode.ROUTER:
        logger.info(f"test result of ROUTER node: {res.prefill_node_rank}")
        assert res.prefill_node_rank == 2, f"expect 2, get {res.prefill_node_rank}"
        res = radix_mesh.match_prefix(k2[:-1])
        logger.info(f"test result of ROUTER node: {res.prefill_node_rank}")
        assert res.prefill_node_rank == 1, f"expect 1, get {res.prefill_node_rank}"
        res = radix_mesh.match_prefix(k2[:-2])
        logger.info(f"test result of ROUTER node: {res.prefill_node_rank}")
        assert res.prefill_node_rank == 0, f"expect 0, get {res.prefill_node_rank}"


if __name__ == '__main__':
    test(sync_and_routing)
    test(multi_write)
