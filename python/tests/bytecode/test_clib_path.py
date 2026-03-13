from pathlib import Path
from unittest.mock import patch

from bloqade.lanes.bytecode._clib_path import has_clib, include_dir, lib_dir, lib_path


class TestIncludeDir:
    def test_returns_path_under_prefix(self):
        with patch("bloqade.lanes.bytecode._clib_path.sys") as mock_sys:
            mock_sys.prefix = "/fake/venv"
            assert include_dir() == Path("/fake/venv/include")


class TestLibDir:
    def test_returns_path_under_prefix(self):
        with patch("bloqade.lanes.bytecode._clib_path.sys") as mock_sys:
            mock_sys.prefix = "/fake/venv"
            assert lib_dir() == Path("/fake/venv/lib")


class TestLibPath:
    def test_darwin(self):
        with patch("bloqade.lanes.bytecode._clib_path.sys") as mock_sys:
            mock_sys.prefix = "/fake/venv"
            mock_sys.platform = "darwin"
            assert lib_path() == Path("/fake/venv/lib/libbloqade_lanes_bytecode.dylib")

    def test_linux(self):
        with patch("bloqade.lanes.bytecode._clib_path.sys") as mock_sys:
            mock_sys.prefix = "/fake/venv"
            mock_sys.platform = "linux"
            assert lib_path() == Path("/fake/venv/lib/libbloqade_lanes_bytecode.so")

    def test_win32(self):
        with patch("bloqade.lanes.bytecode._clib_path.sys") as mock_sys:
            mock_sys.prefix = "/fake/venv"
            mock_sys.platform = "win32"
            assert lib_path() == Path("/fake/venv/lib/bloqade_lanes_bytecode.dll")


class TestHasClib:
    def test_true_when_lib_exists(self, tmp_path):
        lib_file = tmp_path / "lib" / "libbloqade_lanes_bytecode.so"
        lib_file.parent.mkdir()
        lib_file.touch()
        with patch("bloqade.lanes.bytecode._clib_path.sys") as mock_sys:
            mock_sys.prefix = str(tmp_path)
            mock_sys.platform = "linux"
            assert has_clib() is True

    def test_false_when_lib_missing(self, tmp_path):
        with patch("bloqade.lanes.bytecode._clib_path.sys") as mock_sys:
            mock_sys.prefix = str(tmp_path)
            mock_sys.platform = "linux"
            assert has_clib() is False
