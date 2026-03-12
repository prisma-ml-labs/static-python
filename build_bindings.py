#!/usr/bin/env python3
import subprocess
import sys
import os
import shutil
from pathlib import Path

STATIC_DIR = Path(__file__).parent / "static"
BUILD_DIR = STATIC_DIR / "build"


def run(cmd, cwd=None):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def build_bindings():
    if not STATIC_DIR.exists():
        print("Error: static/ submodule not found")
        sys.exit(1)

    if not (STATIC_DIR / "tiktoken-c").exists():
        print("Initializing submodules...")
        run("git submodule update --init --recursive", cwd=STATIC_DIR)

    python_inc = (
        subprocess.check_output("python3-config --includes | cut -d' ' -f1", shell=True)
        .decode()
        .strip()
    )
    pybind11_inc = (
        subprocess.check_output(
            'python3 -c "import pybind11; print(pybind11.get_include())"', shell=True
        )
        .decode()
        .strip()
    )
    python_ldflags = (
        subprocess.check_output("python3-config --ldflags", shell=True).decode().strip()
    )
    ext_suffix = (
        subprocess.check_output("python3-config --extension-suffix", shell=True)
        .decode()
        .strip()
    )

    os.makedirs(BUILD_DIR, exist_ok=True)

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

    output = f"_static{ext_suffix}"
    cmd = (
        f"g++ -std=c++17 -O3 -march=native -ffast-math -Wall -fPIC "
        f"-I{STATIC_DIR} -I{STATIC_DIR}/src -I{STATIC_DIR}/tiktoken-c -I{pybind11_inc} {python_inc} "
        f"-shared -fPIC -o {output} "
        f"{' '.join(str(o) for o in objs)} {STATIC_DIR / 'build' / 'libtiktoken_c.a'} "
        f"pybind.cpp {python_ldflags} -lpthread -ldl"
    )
    run(cmd)
    print(f"Built: {output}")


if __name__ == "__main__":
    build_bindings()
