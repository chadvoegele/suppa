__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import argparse
import hashlib
import logging
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from string import Template

logger = logging.getLogger(__name__)

PROGRAM_NAME = "SUPPA"

# https://github.com/git-lfs/git-lfs/blob/main/docs/spec.md
LFS_HEADER_EXAMPLE = """version 999
oid sha256:10fbdce5d5e2ba7e0249a4a8921faede362fda69bae3c5bb8a59bb1b9407ad5e
size 999999999999999
"""

LFS_HEADER_REGEX = r"""^version (?P<version>[0-9]{1,3})
oid sha256:(?P<oid>[0-9a-f]{64})
size (?P<size>[0-9]{1,15})
"""

LFS_HEADER_TEMPLATE = """version ${version}
oid sha256:${sha256}
size ${size}
"""


def _get_lfs_root():
    variable_name = f"{PROGRAM_NAME}_LFS_ROOT"
    lfs_root = os.getenv(variable_name)

    if lfs_root is None:
        raise NameError(f"Cannot set lfs_root because {variable_name} is not defined!")

    return Path(lfs_root)


def _get_file_size(filename):
    stat_result = os.stat(filename)
    stat_size = stat_result.st_size
    return stat_size


def _get_sha256_hash(filename):
    sha256 = hashlib.new("sha256")
    read_size = 10 * 1024 * 1024
    buffer = None
    with open(filename, "rb") as f:
        while buffer is None or len(buffer) > 0:
            buffer = f.read(read_size)
            sha256.update(buffer)
    sha256_hexdigest = sha256.hexdigest()
    return sha256_hexdigest


@dataclass(frozen=True)
class LfsObject:
    version: int
    oid: str
    size: int
    header_path: str

    @property
    def data_path(self) -> Path:
        lfs_root = _get_lfs_root()
        data_path = lfs_root / self.oid
        return data_path

    def __post_init__(self):
        assert self.version is not None
        assert self.oid is not None
        assert self.size is not None
        assert self.header_path is not None

        if not self.version == 1:
            raise ValueError(f"Only lfs version 1 is supported but found {self.version=}")

        logger.debug(f"Getting size for {self.data_path=}")
        stat_size = _get_file_size(self.data_path)

        if self.size != stat_size:
            raise ValueError(f"Expected size of lfs file to be {self.size=} but was {stat_size=}")

        logger.debug(f"Getting sha256 hash for {self.data_path=}")
        sha256_hexdigest = _get_sha256_hash(self.data_path)

        if sha256_hexdigest != self.oid:
            raise ValueError(f"Expected {self.oid=} from {self.data_path=} but was {sha256_hexdigest=}")

    @classmethod
    def try_from_path(cls, path: Path):
        maximum_lfs_spec_byte_count = len(LFS_HEADER_EXAMPLE)
        with open(path, "r") as f:
            maybe_lfs_header = f.read(maximum_lfs_spec_byte_count)

        re_result = re.search(LFS_HEADER_REGEX, maybe_lfs_header)

        if re_result is None:
            return None

        version = int(re_result.group("version"))
        oid = re_result.group("oid")
        size = int(re_result.group("size"))
        lfs_object = cls(version=version, oid=oid, size=size, header_path=path)
        return lfs_object


@contextmanager
def lfs_open(path, mode="r"):
    """
    https://github.com/python/cpython/blob/3faf8e586d36e73faba13d9b61663afed6a24cb4/Lib/_pyio.py#L1465
    """
    if mode not in ["r", "rb"]:
        with open(path, mode) as f:
            yield f
        return

    maybe_lfs_object = LfsObject.try_from_path(path)

    if maybe_lfs_object is None:
        with open(path, mode) as f:
            yield f
        return

    with open(maybe_lfs_object.data_path, mode) as f:
        yield f


def lfs_add(args):
    path_to_add = Path(args.path_to_add)
    if not path_to_add.exists():
        raise ValueError(f"{path_to_add=} does not exist!")

    logger.debug(f"Processing {path_to_add=}")
    logger.debug(f"Getting file size for {path_to_add=}")
    stat_size = _get_file_size(path_to_add)
    logger.debug(f"Getting sha256 hash for {path_to_add=}")
    sha256_hash = _get_sha256_hash(path_to_add)
    lfs_header = Template(LFS_HEADER_TEMPLATE).substitute(version=1, size=stat_size, sha256=sha256_hash)

    logger.debug(f"Drafted {lfs_header=}")

    lfs_root = _get_lfs_root()
    lfs_path = lfs_root / sha256_hash

    logger.debug(f"Moving {path_to_add} to {lfs_path}")
    shutil.move(path_to_add, lfs_path)
    assert lfs_path.exists()

    with open(path_to_add, "w") as f:
        f.write(lfs_header)

    with open(path_to_add) as f:
        file_lfs_header = f.read()
    logger.debug(f"{path_to_add} now contains {file_lfs_header}")


def main():
    logging.basicConfig()
    logger.setLevel(logging.DEBUG)
    logging.getLogger("suplistml").setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_add", type=str, required=True)

    args = parser.parse_args()
    lfs_add(args)


if __name__ == "__main__":
    main()
