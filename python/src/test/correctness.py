import argparse
import sys

from src.config.cache_config import load_server_args
from src.radix.radix_mesh import RadixMesh


def add_cli_args(parser: argparse.ArgumentParser):
    # Model and port args
    parser.add_argument(
        "--config-file",
        type=str,
        help="The path of the cache nodes",
        required=True,
    )


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_cli_args(parser)
    raw_args = parser.parse_args(sys.argv[1:])
    server_args = load_server_args(raw_args.config_file)
    print(server_args)

    radix_mesh = RadixMesh(server_args)
