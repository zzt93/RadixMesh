import argparse
import sys

from config.cache_config import load_server_args
from test.correctness import add_cli_args
from radix.radix_mesh import RadixMesh


def str_to_list(s: str):
    return [ord(c) for c in s]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    raw_args = parser.parse_args(sys.argv[1:])
    server_args = load_server_args(raw_args.config_file)
    print(server_args)

    p3 = RadixMesh(server_args)
    p2 = RadixMesh(server_args)

    p3.insert(str_to_list('abc'))
    p2.insert(str_to_list('abc'))
    p2.insert(str_to_list('abcd'))
    p2.insert(str_to_list('ab'))
