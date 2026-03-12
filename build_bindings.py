#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"
BUILD_DIR = STATIC_DIR / "build"
PYTHON = sys.executable


def run(cmd, cwd=None):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def get_python_vars():
    import sysconfig

    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    inc = sysconfig.get_path("include")
    ldflags = sysconfig.get_config_var("LDFLAGS")
    return inc, ext_suffix, ldflags


def build_bindings():
    if not STATIC_DIR.exists():
        print("Error: static/ directory not found")
        sys.exit(1)

    if not (STATIC_DIR / "tiktoken-c").exists():
        print("Error: static/tiktoken-c/ directory not found")
        sys.exit(1)

    python_inc, ext_suffix, python_ldflags = get_python_vars()

    pybind11_inc = (
        subprocess.check_output(
            f'{PYTHON} -c "import pybind11; print(pybind11.get_include())"', shell=True
        )
        .decode()
        .strip()
    )

    print(f"Building for Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"Extension suffix: {ext_suffix}")

    os.makedirs(BUILD_DIR, exist_ok=True)
    os.makedirs(Path(__file__).parent / "static", exist_ok=True)

    objs = []
    for src in ["embedder.cpp", "tokenizer_wrapper.cpp", "binary.cpp"]:
        obj = BUILD_DIR / src.replace(".cpp", ".o")
        cmd = f"g++ -std=c++17 -O3 -march=native -ffast-math -Wall -fPIC -I{STATIC_DIR} -I{STATIC_DIR}/tiktoken-c -c -o {obj} {STATIC_DIR}/src/{src}"
        run(cmd)

        if not (
            STATIC_DIR / "tiktoken-c" / "target" / "release" / "libtiktoken_c.a"
        ).exists():
            run("cargo build --release --lib", cwd=STATIC_DIR / "tiktoken-c")

        lib_path = STATIC_DIR / "build" / "libtiktoken_c.a"
        if not lib_path.exists():
            shutil.copy(
                STATIC_DIR / "tiktoken-c" / "target" / "release" / "libtiktoken_c.a",
                lib_path,
            )

        objs.append(obj)

    output = f"static/_static{ext_suffix}"
    cmd = (
        f"g++ -std=c++17 -O3 -march=native -ffast-math -Wall -fPIC "
        f"-I{STATIC_DIR} -I{STATIC_DIR}/src -I{STATIC_DIR}/tiktoken-c -I{pybind11_inc} -I{python_inc} "
        f"-shared -fPIC -o {output} "
        f"{' '.join(str(o) for o in objs)} {STATIC_DIR / 'build' / 'libtiktoken_c.a'} "
        f"pybind.cpp {python_ldflags} -lpthread -ldl"
    )
    run(cmd)
    print(f"Built: {output}")


if __name__ == "__main__":
    build_bindings()
