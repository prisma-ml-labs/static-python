import os
import subprocess
import sys
from pathlib import Path
from setuptools import setup, find_packages


STATIC_DIR = Path(__file__).parent / "static"


def build_bindings():
    if not STATIC_DIR.exists():
        print("Error: static/ submodule not found")
        sys.exit(1)

    if not (STATIC_DIR / "tiktoken-c").exists():
        print("Initializing submodules...")
        subprocess.run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=STATIC_DIR,
            check=True,
        )

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

    build_dir = STATIC_DIR / "build"
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(Path(__file__).parent / "static", exist_ok=True)

    objs = []
    for src in ["embedder.cpp", "tokenizer_wrapper.cpp", "binary.cpp"]:
        obj = build_dir / src.replace(".cpp", ".o")
        cmd = f"g++ -std=c++17 -O3 -march=native -ffast-math -Wall -fPIC -I{STATIC_DIR} -I{STATIC_DIR}/tiktoken-c -c -o {obj} {STATIC_DIR}/src/{src}"
        subprocess.run(cmd, shell=True, check=True)

        if not (
            STATIC_DIR / "tiktoken-c" / "target" / "release" / "libtiktoken_c.a"
        ).exists():
            subprocess.run(
                ["cargo", "build", "--release", "--lib"],
                cwd=STATIC_DIR / "tiktoken-c",
                check=True,
            )

        lib_path = build_dir / "libtiktoken_c.a"
        if not lib_path.exists():
            import shutil

            shutil.copy(
                STATIC_DIR / "tiktoken-c" / "target" / "release" / "libtiktoken_c.a",
                lib_path,
            )

        objs.append(obj)

    output = f"static/_static{ext_suffix}"
    cmd = (
        f"g++ -std=c++17 -O3 -march=native -ffast-math -Wall -fPIC "
        f"-I{STATIC_DIR} -I{STATIC_DIR}/src -I{STATIC_DIR}/tiktoken-c -I{pybind11_inc} {python_inc} "
        f"-shared -fPIC -o {output} "
        f"{' '.join(str(o) for o in objs)} {build_dir / 'libtiktoken_c.a'} "
        f"pybind.cpp {python_ldflags} -lpthread -ldl"
    )
    subprocess.run(cmd, shell=True, check=True)
    print(f"Built: {output}")


def main():
    build_bindings()
    setup(
        name="static-python",
        version="0.1.0",
        packages=find_packages(),
        install_requires=["numpy>=1.20"],
        python_requires=">=3.9",
    )


if __name__ == "__main__":
    main()
