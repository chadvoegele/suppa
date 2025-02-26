__copyright__ = "Copyright (C) 2025 Chad Voegele"
__license__ = "GNU GPLv2"

import os
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from suplistml.data.lfs import lfs_open


class TestLfsOpen(TestCase):
    def test_lfs_open_passthrough(self):
        with NamedTemporaryFile(mode="w+") as f:
            f.write("fake data")
            f.flush()
            with lfs_open(f.name, "r") as g:
                data = g.read()
        self.assertEqual(data, "fake data")

    def test_lfs_open(self):
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tmp_lfs_root = tmpdir / "lfs_root"
            tmp_lfs_root.mkdir()
            with patch.dict(os.environ, {"SUPPA_LFS_ROOT": str(tmp_lfs_root)}):
                fake_dat_path = tmpdir / "fake.dat"
                with open(fake_dat_path, "w") as f:
                    f.write("""version 1
oid sha256:78ad74cecb99d1023206bf2f7d9b11b28767fbb9369daa0afa5e4d062c7ce041
size 10
""")
                fake_hash_path = tmp_lfs_root / "78ad74cecb99d1023206bf2f7d9b11b28767fbb9369daa0afa5e4d062c7ce041"
                with open(fake_hash_path, "w") as f:
                    f.write("fake data\n")

                with lfs_open(fake_dat_path, "r") as g:
                    data = g.read()
            self.assertEqual(data, "fake data\n")
