import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from setuptools import setup
from setuptools.command.build_py import build_py


STATIC_DIR = Path(__file__).parent / "static"
PYTHON = sys.executable


def build_bindings():
    """Compile C++ pybind11 extensions and the tiktoken-c Rust library."""
    if not STATIC_DIR.exists():
        print("Error: static/ directory not found")
        sys.exit(1)

    if not (STATIC_DIR / "tiktoken-c").exists():
        print("Error: static/tiktoken-c/ directory not found")
        sys.exit(1)

    python_inc = sysconfig.get_path("include")
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    python_ldflags = sysconfig.get_config_var("LDFLAGS") or ""

    pybind11_inc = (
        subprocess.check_output(
            f'{PYTHON} -c "import pybind11; print(pybind11.get_include())"',
            shell=True,
        )
        .decode()
        .strip()
    )

    print(f"Building for Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"Extension suffix: {ext_suffix}")

    build_dir = STATIC_DIR / "build"
    os.makedirs(build_dir, exist_ok=True)

    # Build tiktoken-c Rust library first
    tiktoken_lib = STATIC_DIR / "tiktoken-c" / "target" / "release" / "libtiktoken_c.a"
    if not tiktoken_lib.exists():
        subprocess.run(
            ["cargo", "build", "--release", "--lib"],
            cwd=STATIC_DIR / "tiktoken-c",
            check=True,
        )

    lib_path = build_dir / "libtiktoken_c.a"
    if not lib_path.exists():
        shutil.copy(tiktoken_lib, lib_path)

    # Compile C++ object files
    objs = []
    for src in ["embedder.cpp", "tokenizer_wrapper.cpp", "binary.cpp"]:
        obj = build_dir / src.replace(".cpp", ".o")
        cmd = (
            f"g++ -std=c++17 -O3 -march=native -ffast-math -Wall -fPIC "
            f"-I{STATIC_DIR} -I{STATIC_DIR}/tiktoken-c "
            f"-c -o {obj} {STATIC_DIR}/src/{src}"
        )
        subprocess.run(cmd, shell=True, check=True)
        objs.append(obj)

    # Link the shared object
    output = STATIC_DIR / f"_static{ext_suffix}"
    cmd = (
        f"g++ -std=c++17 -O3 -march=native -ffast-math -Wall -fPIC "
        f"-I{STATIC_DIR} -I{STATIC_DIR}/src -I{STATIC_DIR}/tiktoken-c "
        f"-I{pybind11_inc} -I{python_inc} "
        f"-shared -fPIC -o {output} "
        f"{' '.join(str(o) for o in objs)} {lib_path} "
        f"pybind.cpp {python_ldflags} -lpthread -ldl"
    )
    subprocess.run(cmd, shell=True, check=True)
    print(f"Built: {output}")


class BuildPyWithBindings(build_py):
    """Custom build_py that compiles C++ bindings before collecting packages."""

    def run(self):
        build_bindings()
        super().run()


setup(
    cmdclass={"build_py": BuildPyWithBindings},
)
