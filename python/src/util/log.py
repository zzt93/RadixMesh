import logging
import os


def configure_logger(prefix: str = ""):
    if prefix == "":
        prefix = os.getpid()
    format = f"[%(asctime)s][{prefix}] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=format,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
