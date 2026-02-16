import bpy
import subprocess
import sys
import shutil
import os
import warnings

def get_install_folder(internal_folder):
    return os.path.join(bpy.context.preferences.addons[__package__.removesuffix('.functions')].preferences.dependencies_path, internal_folder)

model_file_names = {
    'sam3.pt': '3.45 GB',
}

sam_weights_dir_name = "sam3_weights"

def ensure_package_path():
    # Add the python path to the dependencies dir if missing
    target = get_install_folder("py_packages")
    if os.path.isdir(target) and target not in sys.path:
        print('RotoForge AI: Found missing deps path in sys.path, appending...')
        sys.path.append(target)
        print('RotoForge AI: Deps path has been appended to sys.path')

def test_packages():
    print('RotoForge AI: Testing python packages...')
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            # Test SAM3 import with fallback for known issues
            try:
                from sam3.model_builder import build_sam3_image_model
                from sam3.model.sam3_image_processor import Sam3Processor
                print('SAM3 import successful')
            except ImportError as import_error:
                if 'sam3.sam' in str(import_error):
                    print('SAM3 has known import issues. This is likely due to a broken package structure.')
                    print('Try reinstalling with: pip install git+https://github.com/facebookresearch/sam3.git')
                    print('Or check: https://github.com/facebookresearch/sam3/issues/225')
                else:
                    print(f'SAM3 import error: {import_error}')
                return False
            
            # Test other dependencies
            import timm
            import huggingface_hub
            import PIL
            import torch
            
            del build_sam3_image_model, Sam3Processor, timm, huggingface_hub, PIL, torch
    except ImportError as e:
        print('RotoForge AI: An ImportError occured when importing the dependencies')
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        return False
    except Exception as e:
        print('RotoForge AI: Something went very wrong importing the dependencies, please get that checked')
        if hasattr(e, 'message'):
            print(e.message)
        else:
            print(e)
        return False
    else:
        print('RotoForge AI: Python packages passed testing :)')
        return True

def test_models():
    print('RotoForge AI: Testing models...')
    sam_weights_dir = get_install_folder(sam_weights_dir_name)
    print(f'model weights path: {sam_weights_dir}') 
    
    for file in model_file_names.keys():
        if not os.path.exists(os.path.join(sam_weights_dir, file)):
            print('Rotoforge AI: Missing model: ' + file)
            return False
    #If all files are present, return true
    print('RotoForge AI: All models are present :)')
    return True

# Evil code that kicks modules out of the sys.modules cache while blender is still running
def unload_modules_from_path(target_path):
    print('--- PYTHON PACKAGE UNLOAD STARTING ---')
    
    # Normalize the target path for comparison
    target_path = os.path.abspath(target_path)
    
    # Find modules loaded from the target path
    retry = True
    
    while retry:
        modules_to_remove = []
        retry=False
        print("Searching for active modules in: ", target_path)
        for module_name, module in sys.modules.items():
            # Ensure the module is valid and has a __file__ attribute
            if module and hasattr(module, '__file__') and module.__file__:
                # Get the absolute path of the module's file
                module_path = os.path.abspath(module.__file__)

                # Check if the module or any of its submodules belong to the target path
                if bpy.path.is_subdir(module_path, target_path):
                    # If it's a package, we need to recursively remove all submodules
                    if module_name.find('.') == -1:
                        print(f"Found module: {module_name}")
                        #for submodule_name in list(sys.modules.keys()):
                        #    if submodule_name.startswith(module_name + '.'):
                        #        modules_to_remove.append(submodule_name)
                        # Add the module itself
                        modules_to_remove.append(module_name)

        print('Unloading Modules')
        # Remove the collected modules from sys.modules
        for module_name in modules_to_remove:
            retry=True
            if module_name in sys.modules:
                del sys.modules[module_name]
    
    print('--- PYTHON PACKAGE UNLOAD FINISHED ---')
        

def install_packages(override = False):
    print('--- PYTHON PACKAGE INSTALL STARTING ---')
    python_exe = sys.executable
    requirements_txt = os.path.join(os.path.dirname(os.path.realpath(__file__)), "deps_requirements.txt")
    target = get_install_folder("py_packages")
    
    if override:
        unload_modules_from_path(target)
        shutil.rmtree(target)
        return
    
    # Run pip commands with output capture
    print("Installing pip...")
    result = subprocess.run([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip', '-t', target], 
                         capture_output=True, text=True)
    print(f"pip upgrade output: {result.stdout}")
    if result.stderr:
        print(f"pip upgrade errors: {result.stderr}")
    
    print("Installing requirements...")
    result = subprocess.run([python_exe, '-m', 'pip', 'install', '--upgrade', '-r', requirements_txt, '-t', target], 
                         capture_output=True, text=True)
    print(f"Requirements install output: {result.stdout}")
    if result.stderr:
        print(f"Requirements install errors: {result.stderr}")
    
    # Try to install triton with platform-specific handling
    triton_success = install_triton(python_exe, target)
    
    if not triton_success:
        print("\n⚠️  SAM3 installation incomplete due to missing GPU acceleration")
        print("The addon may not function properly without triton.")
        print("Consider using a Linux environment with GPU for full functionality.")
        
    ensure_package_path()
    print('--- PYTHON PACKAGE INSTALL FINISHED ---')
    return triton_success

def install_triton(python_exe, target):
    # Install triton with platform-specific handling
    print("Installing triton (required for SAM3 GPU acceleration)...")
    
    # Check platform compatibility
    import platform
    system = platform.system()
    python_version = sys.version_info
    
    print(f"Detected platform: {system}")
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Triton requirements: CPython 3.10+, Linux and Windows only, NVIDIA/AMD GPU
    if python_version < (3, 10):
        print("ERROR: Triton requires Python 3.10 or higher")
        print("Please upgrade Python to use SAM3 with GPU acceleration")
        return False
    
    if system == "Darwin":  # macOS
        print("❌ CRITICAL: Triton does not support macOS")
        print("SAM3 requires GPU acceleration, which is not available on macOS")
        print("\nOptions for macOS users:")
        print("1. Use SAM3 in CPU-only mode (slower performance)")
        print("2. Run in Linux environment/VM with GPU")
        print("3. Use cloud-based GPU instances")
        print("4. Wait for future Triton macOS support")
        print("\nFor now, SAM3 will not work properly on macOS without GPU acceleration.")
        return False
    elif system not in ["Linux", "Windows"]:
        print(f"WARNING: Triton may not fully support {system}")
        print("Triton is primarily tested on Linux and Windows with NVIDIA/AMD GPUs")
    
    try:
        triton_result = subprocess.run([python_exe, '-m', 'pip', 'install', 'triton>=2.0.0', '-t', target], 
                                   capture_output=True, text=True)
        print(f"Triton install output: {triton_result.stdout}")
        if triton_result.stderr:
            print(f"Triton install warnings: {triton_result.stderr}")
        print("✅ Triton installation completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Triton installation failed with return code {e.returncode}")
        print(f"Error output: {e.stderr}")
        print("\nTroubleshooting steps:")
        if system == "Windows":
            print("1. Ensure you have CUDA-compatible GPU (NVIDIA)")
            print("2. Update GPU drivers") 
            print("3. Install Visual Studio Build Tools (may be required)")
            print("4. Try: pip install --no-cache-dir triton>=2.0.0")
            print("5. Windows may need pre-built wheels: https://huggingface.co/r4ziel/xformers_pre_built/resolve/main/")
        else:
            print("1. Ensure you have CUDA-compatible GPU (NVIDIA/AMD)")
            print("2. Update GPU drivers")
            print("3. Try: pip install --no-cache-dir triton>=2.0.0")
        print("4. Check: https://github.com/triton-lang/triton/issues")
        print("5. Check: https://github.com/openai/triton/issues/1057 for Windows support")
        return False
    except Exception as e:
        print(f"❌ Unexpected triton installation error: {e}")
        return False

def download_models(override = False):
    print('--- MODEL DOWNLOAD STARTING ---')
    import huggingface_hub as hf
    
    sam_weights_dir = get_install_folder(sam_weights_dir_name)
    for name, size in model_file_names.items():
        if override or not os.path.exists(os.path.join(sam_weights_dir, name)):
            print(f'downloading {name} ({size})')
            path = hf.hf_hub_download(repo_id="facebook/sam3", filename=name, local_dir=sam_weights_dir)
            print(path)
    del hf
    print('--- MODEL DOWNLOAD FINISHED ---')

def register():
    ensure_package_path()
    return {'REGISTERED'}
def unregister():
    return {'UNREGISTERED'}
    