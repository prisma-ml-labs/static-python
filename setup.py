import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from setuptools import setup, find_packages


STATIC_DIR = Path(__file__).parent / "static"
PYTHON = sys.executable


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

    python_inc = sysconfig.get_path("include")
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    python_ldflags = sysconfig.get_config_var("LDFLAGS")

    pybind11_inc = (
        subprocess.check_output(
            f'{PYTHON} -c "import pybind11; print(pybind11.get_include())"', shell=True
        )
        .decode()
        .strip()
    )

    print(f"Building for Python {sys.version_info.major}.{sys.version_info.minor}")
    print(f"Extension suffix: {ext_suffix}")

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
        f"-I{STATIC_DIR} -I{STATIC_DIR}/src -I{STATIC_DIR}/tiktoken-c -I{pybind11_inc} -I{python_inc} "
        f"-shared -fPIC -o {output} "
        f"{' '.join(str(o) for o in objs)} {build_dir / 'libtiktoken_c.a'} "
        f"pybind.cpp {python_ldflags} -lpthread -ldl"
    )
    subprocess.run(cmd, shell=True, check=True)
    print(f"Built: {output}")


def get_static_package_data():
    so_files = list(STATIC_DIR.glob("_static*.so"))
    py_files = list(STATIC_DIR.glob("*.py"))
    return [f.name for f in so_files + py_files]


def main():
    build_bindings()
    setup(
        name="static-python",
        version="0.1.0",
        packages=["static"],
        package_data={"static": get_static_package_data()},
        install_requires=["numpy>=1.20"],
        python_requires=">=3.9",
    )


if __name__ == "__main__":
    main()
