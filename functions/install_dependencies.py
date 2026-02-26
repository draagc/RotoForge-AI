"""
RotoForge AI - Dependency Installation

Sets up a Python 3.12+ virtual environment for the SAM3 inference server.
Blender's own Python needs numpy and Pillow for mask I/O; these are installed
into a local ``blender_packages`` directory on sys.path.
The heavy dependencies (torch, sam3, triton, etc.) go into the venv.
"""

import bpy
import subprocess
import sys
import shutil
import os
import platform


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_install_folder(internal_folder=""):
    base = bpy.context.preferences.addons[
        __package__.removesuffix('.functions')
    ].preferences.dependencies_path
    return os.path.join(base, internal_folder)


def get_blender_packages_dir():
    return get_install_folder("blender_packages")


def ensure_package_path():
    """Add the blender_packages dir to sys.path so Blender can find numpy/Pillow."""
    pkg_dir = get_blender_packages_dir()
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)


def get_venv_dir():
    return get_install_folder("sam3_venv")


def get_venv_python():
    """Return the path to the Python binary inside the venv."""
    venv = get_venv_dir()
    if platform.system() == "Windows":
        return os.path.join(venv, "Scripts", "python.exe")
    return os.path.join(venv, "bin", "python")


def get_server_script():
    """Return the absolute path to sam3_server.py (shipped with the addon)."""
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "sam3_server.py")


# ---------------------------------------------------------------------------
# System Python discovery
# ---------------------------------------------------------------------------

_MIN_PYTHON = (3, 12)

def _find_system_python():
    """Find a system Python >= 3.12.

    Search order:
      1. python3.12, python3.13, python3.14  (explicit version)
      2. python3
      3. python

    Returns the first one that satisfies the version requirement, or None.
    """
    candidates = []
    for minor in range(12, 15):
        candidates.append(f"python3.{minor}")
    candidates += ["python3", "python"]

    for name in candidates:
        exe = shutil.which(name)
        if exe is None:
            continue
        try:
            out = subprocess.run(
                [exe, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode != 0:
                continue
            major, minor = map(int, out.stdout.strip().split())
            if (major, minor) >= _MIN_PYTHON:
                print(f"RotoForge AI: Found suitable Python {major}.{minor} at {exe}")
                return exe
        except Exception:
            continue

    return None


# ---------------------------------------------------------------------------
# Venv creation & package installation
# ---------------------------------------------------------------------------

def create_venv(force=False):
    """Create a Python 3.12+ virtual environment for the SAM3 server.

    Returns True on success, False on failure.
    """
    venv_dir = get_venv_dir()
    venv_python = get_venv_python()

    if os.path.isfile(venv_python) and not force:
        print(f"RotoForge AI: Venv already exists at {venv_dir}")
        return True

    system_python = _find_system_python()
    if system_python is None:
        print("=" * 60)
        print("ERROR: Could not find Python >= 3.12 on your system.")
        print()
        print("SAM3 requires Python 3.12+. Blender's bundled Python is")
        print("too old, so a separate installation is needed.")
        print()
        print("Install Python 3.12+ from https://www.python.org/downloads/")
        print("and make sure it's on your PATH, then try again.")
        print("=" * 60)
        return False

    if force and os.path.isdir(venv_dir):
        print(f"RotoForge AI: Removing old venv at {venv_dir}")
        shutil.rmtree(venv_dir)

    print(f"RotoForge AI: Creating venv at {venv_dir} using {system_python}")
    result = subprocess.run(
        [system_python, "-m", "venv", venv_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"RotoForge AI: venv creation failed:\n{result.stderr}")
        return False

    if not os.path.isfile(venv_python):
        print(f"RotoForge AI: venv Python not found at {venv_python}")
        return False

    print("RotoForge AI: Venv created successfully")
    return True


def _get_blender_python():
    """Return the path to Blender's bundled Python executable.

    sys.executable in Blender points to the Blender binary itself,
    so we need to find the actual Python binary.
    """
    # Blender 4.x+ stores Python at <blender>/Resources/<ver>/python/bin/python3.X
    # Blender also sets sys.prefix to the python dir
    if platform.system() == "Windows":
        candidate = os.path.join(sys.prefix, "bin", "python.exe")
        if not os.path.isfile(candidate):
            candidate = os.path.join(sys.prefix, "python.exe")
    else:
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        candidate = os.path.join(sys.prefix, "bin", py_ver)
        if not os.path.isfile(candidate):
            candidate = os.path.join(sys.prefix, "bin", "python3")

    if os.path.isfile(candidate):
        return candidate

    # Fallback: sys.executable might actually work in some Blender builds
    return sys.executable


def install_blender_packages():
    """Install lightweight packages (numpy, Pillow) into blender_packages/.

    Uses Blender's own Python + pip so the packages are ABI-compatible.
    Returns True on success.
    """
    pkg_dir = get_blender_packages_dir()
    os.makedirs(pkg_dir, exist_ok=True)

    blender_python = _get_blender_python()
    print(f"RotoForge AI: Blender Python at {blender_python}")
    requirements = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "deps_requirements.txt"
    )

    print("RotoForge AI: Installing Blender-side packages...")
    ok = _run_pip(blender_python, [
        "install", "--target", pkg_dir, "-r", requirements
    ])

    if ok:
        ensure_package_path()
        print("RotoForge AI: Blender-side packages installed")
    else:
        print("RotoForge AI: Blender-side package install failed")

    return ok


def install_packages(force=False):
    """Install SAM3 server dependencies into the venv.

    Returns True on success.
    """
    # First: lightweight Blender-side deps
    if not install_blender_packages():
        return False

    print("--- SAM3 SERVER PACKAGE INSTALL STARTING ---")

    if not create_venv(force=force):
        return False

    venv_python = get_venv_python()
    requirements = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "server_requirements.txt"
    )

    # Upgrade pip and ensure setuptools (SAM3 uses pkg_resources)
    print("RotoForge AI: Upgrading pip/setuptools in venv...")
    _run_pip(venv_python, ["install", "--upgrade", "pip", "setuptools"])

    # On Linux/Windows with NVIDIA GPUs, use the CUDA 12.6 wheel index for PyTorch.
    # On macOS, the default PyPI index provides MPS-enabled wheels.
    pip_args = ["install", "-r", requirements]
    if platform.system() != "Darwin":
        pip_args = ["install", "--extra-index-url",
                    "https://download.pytorch.org/whl/cu126",
                    "-r", requirements]

    print("RotoForge AI: Installing server requirements...")
    ok = _run_pip(venv_python, pip_args)

    if ok:
        # Pin setuptools <81 (SAM3 uses deprecated pkg_resources)
        _run_pip(venv_python, ["install", "setuptools<81"])
        _patch_sam3_for_platform(venv_python)
        print("--- SAM3 SERVER PACKAGE INSTALL FINISHED ---")
    else:
        print("--- SAM3 SERVER PACKAGE INSTALL FAILED ---")

    return ok


def _patch_sam3_for_platform(venv_python):
    """Apply compatibility patches to SAM3 source for non-CUDA platforms.

    SAM3 has hard imports of triton (Linux/CUDA only) and decord (no macOS ARM
    wheels) in its module init chain.  We patch the installed SAM3 package
    in-place so it can load on macOS / CPU-only systems.
    """
    import importlib

    venv_dir = get_venv_dir()
    if platform.system() == "Windows":
        site_packages = os.path.join(venv_dir, "Lib", "site-packages")
    else:
        # Find the python version dir
        lib_dir = os.path.join(venv_dir, "lib")
        try:
            py_dirs = [d for d in os.listdir(lib_dir) if d.startswith("python")]
            site_packages = os.path.join(lib_dir, py_dirs[0], "site-packages")
        except (OSError, IndexError):
            print("RotoForge AI: Could not locate venv site-packages for patching")
            return

    sam3_dir = os.path.join(site_packages, "sam3")
    if not os.path.isdir(sam3_dir):
        return

    patches = {
        # 1) Make triton import optional in edt.py
        os.path.join(sam3_dir, "model", "edt.py"): [
            ("import triton\nimport triton.language as tl",
             "try:\n    import triton\n    import triton.language as tl\n"
             "    _HAS_TRITON = True\n"
             "except ImportError:\n    _HAS_TRITON = False\n    triton = None\n    tl = None"),
        ],
        # 2) Make decord import optional
        os.path.join(sam3_dir, "train", "data", "sam3_image_dataset.py"): [
            ("from decord import cpu, VideoReader",
             "try:\n    from decord import cpu, VideoReader\n"
             "except ImportError:\n    cpu = None\n    VideoReader = None"),
        ],
        # 3) Make training-only BatchedDatapoint import optional
        os.path.join(sam3_dir, "model", "sam3_tracker_base.py"): [
            ("from sam3.train.data.collator import BatchedDatapoint",
             "try:\n    from sam3.train.data.collator import BatchedDatapoint\n"
             "except ImportError:\n    BatchedDatapoint = None"),
        ],
        os.path.join(sam3_dir, "model", "sam3_image.py"): [
            ("from sam3.train.data.collator import BatchedDatapoint",
             "try:\n    from sam3.train.data.collator import BatchedDatapoint\n"
             "except ImportError:\n    BatchedDatapoint = None"),
        ],
        # 4) Fix hardcoded device="cuda" in tensor creation (breaks on MPS/CPU)
        os.path.join(sam3_dir, "model", "position_encoding.py"): [
            ('tensors = torch.zeros((1, 1) + size, device="cuda")',
             'tensors = torch.zeros((1, 1) + size, device="cuda" if torch.cuda.is_available() else "cpu")'),
        ],
        os.path.join(sam3_dir, "model", "decoder.py"): [
            ('device="cuda"',
             'device=next(self.parameters()).device'),
        ],
        # 5) Video predictor logging calls torch.cuda APIs directly
        os.path.join(sam3_dir, "model", "sam3_video_predictor.py"): [
            ("torch.cuda.get_arch_list()",
             "(torch.cuda.get_arch_list() if torch.cuda.is_available() else [])"),
            ("torch.cuda.get_device_properties(torch.cuda.current_device())",
             "('MPS' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'CPU')"),
        ],
        # 6) Video predictor: use single-GPU class on non-CUDA systems
        os.path.join(sam3_dir, "model_builder.py"): [
            ("from sam3.model.sam3_video_predictor import Sam3VideoPredictorMultiGPU",
             "from sam3.model.sam3_video_predictor import Sam3VideoPredictor, Sam3VideoPredictorMultiGPU"),
            ("def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):\n"
             "    return Sam3VideoPredictorMultiGPU(",
             "def build_sam3_video_predictor(*model_args, gpus_to_use=None, **model_kwargs):\n"
             "    if not torch.cuda.is_available():\n"
             "        return Sam3VideoPredictor(*model_args, **model_kwargs)\n"
             "    return Sam3VideoPredictorMultiGPU("),
        ],
    }

    # Also replace the edt.py with our fallback version if triton isn't available
    edt_fallback = os.path.join(os.path.dirname(os.path.realpath(__file__)), "edt_fallback.py")
    edt_target = os.path.join(sam3_dir, "model", "edt.py")

    patched_count = 0
    for filepath, replacements in patches.items():
        if not os.path.isfile(filepath):
            continue
        try:
            with open(filepath, "r") as f:
                content = f.read()
            changed = False
            for old, new in replacements:
                # Skip if already patched (the replacement text is present)
                if new in content:
                    continue
                if old in content:
                    content = content.replace(old, new, 1)
                    changed = True
            if changed:
                with open(filepath, "w") as f:
                    f.write(content)
                patched_count += 1
        except Exception as e:
            print(f"RotoForge AI: Warning — failed to patch {filepath}: {e}")

    # Copy the full edt fallback that has a proper PyTorch-only implementation
    if os.path.isfile(edt_fallback):
        try:
            shutil.copy2(edt_fallback, edt_target)
            patched_count += 1
        except Exception as e:
            print(f"RotoForge AI: Warning — failed to copy EDT fallback: {e}")

    if patched_count:
        print(f"RotoForge AI: Applied {patched_count} platform compatibility patches to SAM3")


def _run_pip(python_exe, pip_args):
    """Run a pip command inside the venv and print output."""
    cmd = [python_exe, "-m", "pip"] + pip_args
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout:
        for line in result.stdout.strip().split("\n")[-10:]:
            print(f"  {line}")
    if result.returncode != 0:
        print(f"  pip error (code {result.returncode}):")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-15:]:
                print(f"  {line}")
        return False
    return True


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

def test_venv():
    """Check that the venv exists and can import SAM3."""
    print("RotoForge AI: Testing venv...")
    venv_python = get_venv_python()
    if not os.path.isfile(venv_python):
        print("RotoForge AI: Venv Python not found")
        return False

    result = subprocess.run(
        [venv_python, "-c",
         "import torch; import sam3; from sam3.model_builder import build_sam3_image_model; "
         "print('OK', torch.__version__)"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        print(f"RotoForge AI: Venv import test failed:\n{result.stderr}")
        return False

    print(f"RotoForge AI: Venv OK — {result.stdout.strip()}")
    return True


def test_blender_packages():
    """Check that Blender's Python has the lightweight deps we need."""
    print("RotoForge AI: Testing Blender-side packages...")
    ensure_package_path()
    try:
        import numpy
        import PIL.Image
        print("RotoForge AI: Blender-side packages OK")
        return True
    except ImportError as e:
        print(f"RotoForge AI: Missing Blender-side package: {e}")
        return False


# ---------------------------------------------------------------------------
# Register / unregister
# ---------------------------------------------------------------------------

def register():
    try:
        ensure_package_path()
    except (KeyError, AttributeError):
        pass
    return {'REGISTERED'}

def unregister():
    return {'UNREGISTERED'}
